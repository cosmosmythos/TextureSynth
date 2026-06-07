#include "engine/Engine.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeRegistryLoader.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <climits>
#include <thread>
#include <unordered_set>

namespace te {

// Specialization helper. Builds VkSpecializationInfo from variant key constants. Returns nullptr when specialization_count == 0. Data lives only for create call duration (Vulkan reads synchronously).
static const VkSpecializationInfo* build_spec_info(const ShaderVariantKey& vk,
                                                   VkSpecializationMapEntry (&out_entries)[8],
                                                   VkSpecializationInfo&   out_info) {
    if (vk.specialization_count == 0) return nullptr;
    const uint32_t n = vk.specialization_count > 8 ? 8 : vk.specialization_count;
    for (uint32_t i = 0; i < n; ++i) {
        out_entries[i].constantID = i;
        out_entries[i].offset     = i * sizeof(uint32_t);
        out_entries[i].size       = sizeof(uint32_t);
    }
    out_info.mapEntryCount = n;
    out_info.pMapEntries   = out_entries;
    out_info.dataSize      = n * sizeof(uint32_t);
    out_info.pData         = vk.specialization.data();
    return &out_info;
}

// Lifecycle guard. Top of every public mutator that requires Ready. Holds entry_mu_ for function body. Concurrent shutdown/init blocks until method returns.
#define TE_GUARD_READY(call)                                                     \
    std::lock_guard<std::recursive_mutex> te_lk(entry_mu_);                       \
    if (state_.load(std::memory_order_acquire) != EngineState::Ready) {          \
        set_error_(EngineErrorCode::UseAfterShutdown,                            \
                   std::string("engine not Ready (state=")                       \
                   + std::to_string((int)state_.load()) + ")",                   \
                   EnginePhase::Idle);                                           \
        return call;                                                             \
    }

// init. Holds entry_mu_ for entire body. Concurrent shutdown blocks until init finishes. State machine: Uninitialized/ShutDown/Error -> Initializing -> Ready (or -> Error on failure).
bool Engine::init(VkSurfaceKHR surface,
                  const char** extra_inst_exts, uint32_t extra_inst_ext_count,
                  bool enable_validation,
                  const std::string& cache_dir,
                  const std::string& nodes_dir,
                  const std::string& glsl_dir) {

    // entry_mu_ held for the entire body. A concurrent Engine::shutdown
    // will block here until this function returns.
    std::lock_guard<std::recursive_mutex> lk(entry_mu_);

    const EngineState s = state_.load(std::memory_order_acquire);
    if (s == EngineState::Initializing || s == EngineState::ShuttingDown) {
        // Unreachable while we hold the lock: only init() sets Initializing,
        // only shutdown() sets ShuttingDown, and both take this same lock.
        set_error_(EngineErrorCode::Busy,
                   "engine init or shutdown already in progress",
                   EnginePhase::Init);
        return false;
    }
    if (s == EngineState::Ready) {
        log_info("Engine::init: already Ready -- no-op (idempotent)");
        return true;
    }
    if (s == EngineState::ShutDown) {
        log_info("Engine::init: re-arming after ShutDown");
    }
    state_.store(EngineState::Initializing, std::memory_order_release);

    clear_error();
    bool did_partially_succeed = false;
    std::string err;  // hoisted to dodge MSVC C2362 (goto past init)

    VulkanContextDesc d{};
    d.enable_validation = enable_validation;
    d.surface = surface;
    d.extra_instance_extensions = extra_inst_exts;
    d.extra_instance_extension_count = extra_inst_ext_count;
    if (!ctx_.init(d)) {
        set_error_(EngineErrorCode::InitFailed,
                   "Vulkan context init failed (no compatible device or surface?)",
                   EnginePhase::Init);
        goto rollback;
    }
    did_partially_succeed = true;  // first Vulkan object exists; rollback may need to tear it down

    cache_ = std::make_unique<ShaderCache>(cache_dir);

    // Loading zero nodes is fatal: a node-less engine cannot compile any graph. Refuse to come up so failure is visible at launch.
    if (NodeRegistryLoader::load_from_directory(node_lib_, nodes_dir, glsl_dir, &err) == 0) {
        set_error_(EngineErrorCode::InitFailed,
                   "no nodes loaded from '" + nodes_dir + "': " + err,
                   EnginePhase::Init);
        goto rollback;
    }

    output_storage_ = std::make_unique<Image>();
    if (!output_storage_->create(ctx_, output_w_, output_h_)) {
        set_error_(EngineErrorCode::InitFailed,
                   "output storage image creation failed",
                   EnginePhase::Init);
        goto rollback;
    }
    if (!async_.init(ctx_, 4096, 4096)) {
        set_error_(EngineErrorCode::InitFailed,
                   "AsyncReadback init failed",
                   EnginePhase::Init);
        goto rollback;
    }
    if (!uploader_.init(ctx_)) {
        set_error_(EngineErrorCode::InitFailed,
                   "ImageUploader init failed",
                   EnginePhase::Init);
        goto rollback;
    }
    if (!ensure_dummy_image_()) {
        set_error_(EngineErrorCode::InitFailed,
                   "dummy image creation failed",
                   EnginePhase::Init);
        goto rollback;
    }

    // ── Create param SSBO ring buffers (host-visible, persistently mapped) ──
    {
        const VkDeviceSize sz = (VkDeviceSize)MAX_NODE_PARAMS * sizeof(float);
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size  = sz;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
                  | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        for (uint32_t i = 0; i < PARAM_RING; ++i) {
            VmaAllocationInfo ai{};
            if (vmaCreateBuffer(ctx_.allocator(), &bci, &aci,
                                &param_buf_[i], &param_alloc_[i], &ai) != VK_SUCCESS) {
                set_error_(EngineErrorCode::InitFailed,
                           "param SSBO alloc failed (VMA out of memory?)",
                           EnginePhase::Init);
                goto rollback;
            }
            const std::string ring_name = "param_ssbo_ring[" + std::to_string(i) + "]";
            ctx_.set_debug_name(VK_OBJECT_TYPE_BUFFER, (uint64_t)param_buf_[i], ring_name);
            // VMA-side name (queryable via vmaGetAllocationInfo().pName).
            // See ResourceManager::create_image_ for the rationale.
            vmaSetAllocationName(ctx_.allocator(), param_alloc_[i], ring_name.c_str());
            param_mapped_[i] = ai.pMappedData;
            std::memset(param_mapped_[i], 0, (size_t)sz);
            vmaFlushAllocation(ctx_.allocator(), param_alloc_[i], 0, VK_WHOLE_SIZE);
        }
    }

    if (!create_global_samplers_()) {
        set_error_(EngineErrorCode::InitFailed,
                   "global sampler creation failed",
                   EnginePhase::Init);
        goto rollback;
    }

    // Bindless table — ONE descriptor set for the whole engine lifetime.
    if (!bindless_.init(ctx_, sampler_repeat_, sampler_clamp_, sampler_mirror_,
                        sizeof(PassPushConstants))) {
        set_error_(EngineErrorCode::InitFailed,
                   "BindlessTable init failed",
                   EnginePhase::Init);
        goto rollback;
    }

    // Reserved slot 0 in BindlessTable is the dummy. Write the dummy view there.
    dummy_sampled_slot_ = 0;
    bindless_.write_sampled(ctx_, dummy_sampled_slot_, dummy_.view(), VK_IMAGE_LAYOUT_GENERAL);

    // ── Param SSBO ring: write all PARAM_RING buffers ONCE. Never rebound. ──
    {
        std::array<VkBuffer, BindlessTable::PARAM_RING_SIZE> ring_bufs{};
        for (uint32_t i = 0; i < PARAM_RING; ++i) ring_bufs[i] = param_buf_[i];
        bindless_.write_param_ring(ctx_, ring_bufs, MAX_NODE_PARAMS * sizeof(float));
    }
    param_write_idx_ = 0;
    state_.store(EngineState::Ready, std::memory_order_release);
    return true;

rollback:
    if (did_partially_succeed) {
        shutdown_internal_();
    }
    state_.store(EngineState::Error, std::memory_order_release);
    return false;
}


bool Engine::ensure_dummy_image_() {
    return dummy_.create(ctx_, 1, 1);
}


// shutdown. Idempotent. Holds entry_mu_ for entire teardown. Tolerates VK_ERROR_DEVICE_LOST. State: Uninitialized -> ShutDown, Ready/Error -> ShuttingDown -> ShutDown.
void Engine::shutdown() {
    std::lock_guard<std::recursive_mutex> lk(entry_mu_);

    const EngineState s = state_.load(std::memory_order_acquire);
    if (s == EngineState::ShutDown) return;
    if (s == EngineState::Uninitialized) {
        state_.store(EngineState::ShutDown, std::memory_order_release);
        return;
    }
    state_.store(EngineState::ShuttingDown, std::memory_order_release);
    shutdown_internal_();
    state_.store(EngineState::ShutDown, std::memory_order_release);
}


void Engine::shutdown_internal_() {
    // Wait for the GPU to finish. Tolerate device-lost so a sick GPU still allows cleanup.
    if (ctx_.device()) {
        const VkResult w = vkDeviceWaitIdle(ctx_.device());
        if (w != VK_SUCCESS && w != VK_ERROR_DEVICE_LOST) {
            log_warn("Engine::shutdown_internal_: vkDeviceWaitIdle returned "
                     + std::to_string(w));
        }
    }
    async_.drain(ctx_);
    uploader_.drain(ctx_);

    for (auto& pp : pending_passes_) if (pp.fut.valid()) pp.fut.wait();
    pending_passes_.clear();

    for (auto& p : passes_) {
        if (p.pipeline) p.pipeline->destroy(ctx_);
    }
    passes_.clear();
    for (auto& r : retired_passes_) {
        if (r.pipeline) r.pipeline->destroy(ctx_);
    }
    retired_passes_.clear();

    for (auto& ce : chain_execs_) {
        if (ce.pipeline) ce.pipeline->destroy(ctx_);
    }
    chain_execs_.clear();
    chain_id_of_pass_.clear();

    if (dummy_sampled_slot_ != BindlessTable::INVALID_SLOT) {
        dummy_sampled_slot_ = BindlessTable::INVALID_SLOT;
    }
    // Release external image slots.
    for (auto& kv : ext_sampled_slot_) bindless_.free_sampled_slot(kv.second);
    ext_sampled_slot_.clear();
    // Resource slots are flushed when ResourceManager retires.
    res_sampled_slot_.clear();
    res_storage_slot_.clear();

    bindless_.shutdown(ctx_);
    destroy_global_samplers_();

    for (uint32_t i = 0; i < PARAM_RING; ++i) {
        if (param_buf_[i]) {
            vmaDestroyBuffer(ctx_.allocator(), param_buf_[i], param_alloc_[i]);
            param_buf_[i] = VK_NULL_HANDLE;
            param_alloc_[i] = nullptr;
            param_mapped_[i] = nullptr;
        }
    }

    resources_.shutdown(ctx_);
    for (auto& kv : image_registry_) if (kv.second) kv.second->destroy(ctx_);
    image_registry_.clear();

    dummy_.destroy(ctx_);
    uploader_.shutdown(ctx_);
    async_.shutdown(ctx_);
    for (auto& r : retired_images_) if (r.img) r.img->destroy(ctx_);
    retired_images_.clear();
    if (output_storage_) { output_storage_->destroy(ctx_); output_storage_.reset(); }
    ctx_.shutdown();
}


bool Engine::create_global_samplers_() {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.mipLodBias = 0.0f;
    sci.anisotropyEnable = VK_FALSE;
    sci.compareEnable = VK_FALSE;
    sci.minLod = 0.0f;
    sci.maxLod = 1.0f;
    sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    sci.unnormalizedCoordinates = VK_FALSE;

    // Repeat
    if (vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_repeat_) != VK_SUCCESS) {
        set_error_(EngineErrorCode::InitFailed, "vkCreateSampler(repeat) failed",
                   EnginePhase::Init);
        return false;
    }

    // Clamp
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_clamp_) != VK_SUCCESS) {
        set_error_(EngineErrorCode::InitFailed, "vkCreateSampler(clamp) failed",
                   EnginePhase::Init);
        return false;
    }

    // Mirror
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    if (vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_mirror_) != VK_SUCCESS) {
        set_error_(EngineErrorCode::InitFailed, "vkCreateSampler(mirror) failed",
                   EnginePhase::Init);
        return false;
    }

    return true;
}


void Engine::destroy_global_samplers_() {
    if (sampler_repeat_) { vkDestroySampler(ctx_.device(), sampler_repeat_, nullptr); sampler_repeat_ = VK_NULL_HANDLE; }
    if (sampler_clamp_)  { vkDestroySampler(ctx_.device(), sampler_clamp_, nullptr);  sampler_clamp_  = VK_NULL_HANDLE; }
    if (sampler_mirror_) { vkDestroySampler(ctx_.device(), sampler_mirror_, nullptr); sampler_mirror_ = VK_NULL_HANDLE; }
}


void Engine::assign_bindless_slots_(PassExec& pe) {

    // Outputs
    pe.output_count = static_cast<uint32_t>(pe.output_resources.size());
    for (uint32_t t = 0; t < pe.output_count && t < MAX_PASS_OUTPUTS; ++t) {
        const ResourceUUID& rid = pe.output_resources[t];
        auto it_o = res_storage_slot_.find(rid);
        if (it_o == res_storage_slot_.end()) {
            uint32_t slot = bindless_.alloc_storage_slot();
            if (slot == BindlessTable::INVALID_SLOT) {
                log_error("Bindless storage exhausted");
                slot = 0;
            }
            else {
                res_storage_slot_[rid] = slot;
                if (auto* r = resources_.get(rid))
                    bindless_.write_storage(ctx_, slot, r->view);
            }
            pe.out_storage_slots[t] = slot;
        }
        else {
            pe.out_storage_slots[t] = it_o->second;
        }
    }


    // Inputs: sampled slot per input resource.
    pe.input_count = static_cast<uint32_t>(pe.input_resources.size());
    for (uint32_t i = 0; i < pe.input_count && i < MAX_PASS_INPUTS; ++i) {
        const ResourceUUID rid = pe.input_resources[i];
        uint32_t slot = BindlessTable::INVALID_SLOT;

        if (rid.node_id == 0) {
            auto eit = image_registry_.find(pe.node_id);
            if (eit != image_registry_.end() && eit->second) {
                auto sit = ext_sampled_slot_.find(pe.node_id);
                if (sit == ext_sampled_slot_.end()) {
                    slot = bindless_.alloc_sampled_slot();
                    if (slot == BindlessTable::INVALID_SLOT) slot = dummy_sampled_slot_;
                    else {
                        ext_sampled_slot_[pe.node_id] = slot;
                        bindless_.write_sampled(ctx_, slot, eit->second->view(),
                                                VK_IMAGE_LAYOUT_GENERAL);
                    }
                } else slot = sit->second;
            } else slot = dummy_sampled_slot_;
        } else {
            auto sit = res_sampled_slot_.find(rid);
            if (sit == res_sampled_slot_.end()) {
                slot = bindless_.alloc_sampled_slot();
                if (slot == BindlessTable::INVALID_SLOT) slot = dummy_sampled_slot_;
                else {
                    res_sampled_slot_[rid] = slot;
                    if (auto* r = resources_.get(rid))
                        bindless_.write_sampled(ctx_, slot, r->view, VK_IMAGE_LAYOUT_GENERAL);
                }
            } else slot = sit->second;
        }
        pe.in_sampled_slots[i] = slot;
    }
    for (uint32_t i = pe.input_count; i < MAX_PASS_INPUTS; ++i)
        pe.in_sampled_slots[i] = dummy_sampled_slot_;

}


bool Engine::create_pass_pipeline_(PassExec& pe,
                                   NodeId node_id,
                                   const std::string& type_id,
                                   const std::string& error_prefix,
                                   const ShaderVariantKey& variant_key,
                                   const std::vector<uint32_t>& spirv) {
    auto pipe = std::make_unique<ComputePipeline>();
    const std::string pipe_name = "pipe_node_" + std::to_string(node_id) + "_" + type_id;
    pipe->set_name(pipe_name);
    // VkSpecializationInfo is non-extensible; value-init only (no sType).
    VkSpecializationMapEntry entries[8];
    VkSpecializationInfo     spec{};
    const VkSpecializationInfo* spec_ptr = build_spec_info(variant_key, entries, spec);
    if (!pipe->create(ctx_, spirv, bindless_.pipeline_layout(), spec_ptr)) {
        set_error_(EngineErrorCode::PipelineCreation, error_prefix,
                   EnginePhase::GraphCompileFinish, node_id);
        return false;
    }
    pe.pipeline = std::move(pipe);
    ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                        (uint64_t)pe.pipeline->pipeline(),
                        pe.pipeline->name());
    return true;
}


uint64_t Engine::set_graph(const Graph& graph) {
    TE_GUARD_READY(0);
    clear_error();

    // Topology change must drain in-flight async work before we tear down
    // descriptor sets, pipelines, or per-node images.
    async_.drain(ctx_);

    auto ir_result = validate_graph(graph, node_lib_);
    if (!ir_result.success) {
        set_error_(EngineErrorCode::GraphValidation, ir_result.error,
                   EnginePhase::GraphSubmit);
        return 0;
    }

    auto compile_result = GraphCompiler::compile(ir_result.ir, node_lib_);
    if (!compile_result.success) {
        set_error_(EngineErrorCode::GraphCompile, compile_result.error,
                   EnginePhase::GraphSubmit);
        return 0;
    }

    current_graph_ = graph;
    current_ir_    = std::move(ir_result.ir);
    current_revision_++;
    rebuild_downstream_adj_();
    any_pass_dirty_.store(true);
    dirty_set_.mark_topology_change();
    last_presented_pixels_.clear();
    last_presented_w_ = 0;
    last_presented_h_ = 0;

    // Retire old pass executables FIRST so their descriptor sets stop
    // referencing images that ResourceManager is about to recycle.
    retire_all_passes_();
    // Flush all resource-bound bindless slots (they pointed at the old graph's images).
    for (auto& kv : res_sampled_slot_) bindless_.free_sampled_slot(kv.second);
    for (auto& kv : res_storage_slot_) bindless_.free_storage_slot(kv.second);
    res_sampled_slot_.clear();
    res_storage_slot_.clear();

    std::string rerr;
    if (!resources_.allocate_for_graph(ctx_, current_ir_, node_lib_, output_w_, output_h_,
                                        texture_format_, &rerr,
                                        &compile_result.pass_plan.color_classes)) {
        set_error_(EngineErrorCode::GraphCompile, rerr, EnginePhase::GraphSubmit);
        return 0;
    }

    param_base_slot_     = std::move(compile_result.param_base_slot);
    total_param_floats_  = compile_result.total_param_floats;

    for (uint32_t i = 0; i < PARAM_RING; ++i) {
        if (param_mapped_[i]) {
            std::memset(param_mapped_[i], 0, MAX_NODE_PARAMS * sizeof(float));
            vmaFlushAllocation(ctx_.allocator(), param_alloc_[i], 0, VK_WHOLE_SIZE);
        }
    }
    // Seed param SSBOs with manifest defaults so nodes render correctly
    for (uint32_t ring = 0; ring < PARAM_RING; ++ring) {
        auto* dst = static_cast<float*>(param_mapped_[ring]);
        if (!dst) continue;

        for (const auto& vn : current_ir_.nodes) {
            auto bit = param_base_slot_.find(vn.id);
            if (bit == param_base_slot_.end()) continue;

            const auto* type = node_lib_.find(vn.type_id);
            if (!type) continue;

            const int base = bit->second;
            for (size_t i = 0; i < type->params.size(); ++i) {
                const int slot = base + static_cast<int>(i);
                if (slot >= 0 && slot < static_cast<int>(MAX_NODE_PARAMS)) {
                    float v = type->params[i].default_value;
                    if (!std::isfinite(v)) v = 0.0f;
                    dst[slot] = v;
                }
            }
        }

        vmaFlushAllocation(ctx_.allocator(), param_alloc_[ring], 0, VK_WHOLE_SIZE);
    }
    param_write_idx_ = 0;

    const uint64_t gen = ++compile_generation_;

    for (auto& pp : pending_passes_) if (pp.fut.valid()) pp.fut.wait();
    pending_passes_.clear();

    bool all_cached = true;
    std::vector<std::vector<uint32_t>> cached_spv(compile_result.pass_plan.passes.size()); // sparse: pre-size, write by index
    for (size_t i = 0; i < compile_result.pass_plan.passes.size(); ++i) {
        const auto& pass = compile_result.pass_plan.passes[i];
        if (pass.kind != PassKind::Dispatch) continue;
        auto blob = cache_->load(pass.variant_key);
        if (!blob) { all_cached = false; break; }
        cached_spv[i] = std::move(*blob);  // sparse write: index by pass position
    }

    if (all_cached) {
        passes_.clear();
        passes_.reserve(compile_result.pass_plan.passes.size());

        for (size_t i = 0; i < compile_result.pass_plan.passes.size(); ++i) {
            const auto& pass = compile_result.pass_plan.passes[i];
            PassExec pe;
            pe.node_id          = pass.node_id;
            pe.output_resources = pass.output_resources;
            pe.input_resources  = pass.input_resources;
            pe.param_base_slot  = pass.param_base_slot;
            pe.output_layout_is_general = false;
            pe.bypassed         = pass.bypassed;

            if (pass.kind == PassKind::Dispatch) {
                // Bypassed passes: no pipeline; executor clears output to zero.
                // We still assign bindless slots so downstream passes can resolve.
                if (!pass.bypassed) {
                    if (!create_pass_pipeline_(pe, pass.node_id, pass.type_id,
                            "pipeline creation failed for node " + std::to_string(pass.node_id),
                            pass.variant_key, cached_spv[i])) {
                        return 0;
                    }
                }
                assign_bindless_slots_(pe);
            }
            passes_.push_back(std::move(pe));
        }
        final_output_resource_ = compile_result.pass_plan.final_output_resource;
        installed_generation_  = gen;
        populate_chains_(compile_result.pass_plan);
        log_info("PassPlan installed (cache hit), passes=" + std::to_string(passes_.size())
                 + " generation=" + std::to_string(gen));
        return gen;
    }

    pending_passes_.clear();
    pending_passes_.reserve(compile_result.pass_plan.passes.size());
    for (auto& pass : compile_result.pass_plan.passes) {
        PendingPass pp;
        pp.name             = "node_" + std::to_string(pass.node_id);
        pp.type_id          = pass.type_id;
        pp.input_count      = pass.input_socket_count;
        pp.output_resources = pass.output_resources;
        pp.input_resources  = pass.input_resources;
        pp.node_id          = pass.node_id;
        pp.kind             = pass.kind;
        pp.variant_key      = pass.variant_key;
        pp.bypassed         = pass.bypassed;
        if (pass.kind == PassKind::Dispatch && !pass.bypassed) {
            // Phase 1c: bypassed passes skip the async shader compile —
            // the executor clears their output to zero.
            pp.fut = compiler_.compile_compute_async(pass.shader_glsl, pp.name);
        }
        pending_passes_.push_back(std::move(pp));
    }
    pending_active_       = true;
    pending_generation_   = gen;
    pending_final_output_ = compile_result.pass_plan.final_output_resource;
    pending_pass_plan_    = compile_result.pass_plan;
    log_info("PassPlan compile dispatched, passes=" + std::to_string(pending_passes_.size())
             + " generation=" + std::to_string(gen));
    return gen;
}


uint64_t Engine::set_active_node(NodeId node_id) {
    if (current_graph_.nodes.empty()) {
        set_error_(EngineErrorCode::GraphValidation,
                   "set_active_node called before any set_graph", EnginePhase::GraphSubmit);
        return 0;
    }
    if (node_id == current_graph_.output_node) {
        return installed_generation_ ? installed_generation_ : pending_generation_;
    }
    Graph g = current_graph_;
    g.output_node = node_id;
    return set_graph(g);
}


Engine::BakedImage Engine::readback_sync() {
    BakedImage out;
    if (current_graph_.nodes.empty()) {
        set_error_(EngineErrorCode::GraphValidation,
                   "readback_sync called before any set_graph", EnginePhase::Submit);
        return out;
    }
    if (!has_pipeline()) {
        for (int i = 0; i < 200 && !has_pipeline(); ++i) {
            poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (!has_pipeline()) {
            set_error_(EngineErrorCode::CompileFailed,
                       "readback_sync: compile timed out", EnginePhase::Submit);
            return out;
        }
    }
    PushConstants pc{};
    pc.resolution_x = output_w_;
    pc.resolution_y = output_h_;
    pc.seed         = 1;
    pc.time         = 0.0f;

    async_.drain(ctx_);
    uint64_t gen = async_.submit(ctx_, *this, pc, installed_generation_);
    if (gen == 0) {
        set_error_(EngineErrorCode::SubmitFailed,
                   "readback_sync: async submit failed", EnginePhase::Submit);
        return out;
    }
    uint64_t og = 0;
    for (int i = 0; i < 4000; ++i) {
        if (async_.poll(ctx_, out.pixels, out.width, out.height, og)) return out;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    set_error_(EngineErrorCode::ReadbackFailed,
               "readback_sync: readback timed out", EnginePhase::Submit);
    return out;
}


std::vector<Engine::BakedImage> Engine::bake() {
    std::vector<BakedImage> out;
    if (current_graph_.nodes.empty()) {
        set_error_(EngineErrorCode::GraphValidation,
                   "bake called before any set_graph", EnginePhase::GraphSubmit);
        return out;
    }

    struct T { NodeId node; std::string name; };
    std::vector<T> targets;
    if (current_graph_.output_targets.empty()) {
        targets.push_back({current_graph_.output_node, "output"});
    } else {
        for (const auto& t : current_graph_.output_targets) {
            targets.push_back({t.source_node, t.name});
        }
    }

    const NodeId original_active = current_graph_.output_node;
    for (const auto& t : targets) {
        if (t.node != current_graph_.output_node) {
            uint64_t g = set_active_node(t.node);
            if (g == 0) return out;
        }
        BakedImage img = readback_sync();
        if (img.pixels.empty()) return out;
        img.name = t.name;
        out.push_back(std::move(img));
    }
    if (current_graph_.output_node != original_active) {
        set_active_node(original_active);
    }
    return out;
}


// rebuild_downstream_adj_ -- build adjacency list for dirty propagation
void Engine::rebuild_downstream_adj_() {
    downstream_adj_.clear();
    for (const auto& vn : current_ir_.nodes) {
        downstream_adj_[vn.id];   // ensure key exists even for sinks
    }
    for (const auto& c : current_ir_.connections) {
        downstream_adj_[c.src_node].push_back(c.dst_node);
    }
}


// Stage 6: check aliased resources for memory staleness. If a resource is
// clean (producer not dirty this frame) but its alias_gen < group.current_gen
// (another group member wrote to the shared memory since), mark the producer
// dirty so the per-pass loop re-writes it. Safe to call during recording
// because dirty_set_ is consumed before any dispatch happens.
bool Engine::resolve_aliased_staleness_() {
    bool found_stale = false;
    for (auto& kv : resources_.live_resources()) {
        const auto& r = kv.second;
        if (r.alias_group_id == 0) continue;
        const auto& pools = resources_.alias_pools();
        if (r.alias_group_id > pools.size()) continue;
        uint64_t gen = pools[r.alias_group_id - 1].current_gen;
        if (r.alias_gen < gen && !dirty_set_.is_dirty(r.node_id)) {
            mark_node_dirty(r.node_id);
            found_stale = true;
        }
    }
    return found_stale;
}


void Engine::poll_pending_compiles() {
    TE_GUARD_READY(;);

    // Pull completed uploads. Rewire descriptors and mark dirty.
    auto completed = uploader_.poll(ctx_);
    for (auto& c : completed) {
        image_registry_[c.node_id] = std::move(c.image);
        images_needing_acquire_[c.node_id] = (ctx_.has_dedicated_transfer());
        pending_uploads_.erase(
            std::remove_if(pending_uploads_.begin(), pending_uploads_.end(),
                [&](const PendingUpload& p){ return p.ticket == c.ticket; }),
            pending_uploads_.end());

        // Rewrite the bindless sampled slot for this external image.
        auto sit = ext_sampled_slot_.find(c.node_id);
        uint32_t slot = (sit == ext_sampled_slot_.end())
                      ? bindless_.alloc_sampled_slot()
                      : sit->second;
        ext_sampled_slot_[c.node_id] = slot;
        if (auto* img = image_registry_[c.node_id].get())
            bindless_.write_sampled(ctx_, slot, img->view(), VK_IMAGE_LAYOUT_GENERAL);

        // Patch any pass on this node to point at the new slot.
        for (auto& pe : passes_) {
            if (pe.node_id != c.node_id) continue;
            for (uint32_t i = 0; i < pe.input_count && i < MAX_PASS_INPUTS; ++i)
                if (pe.input_resources[i].node_id == 0)
                    pe.in_sampled_slots[i] = slot;
        }
        mark_downstream_dirty_(c.node_id);
    }

    tick_retired();
    resources_.tick(ctx_);

    if (!pending_active_) return;

    // Wait until every future is ready (non-blocking peek).
    for (auto& pp : pending_passes_) {
        if (pp.kind != PassKind::Dispatch) continue;
        if (pp.bypassed) continue;             // Phase 1c: no shader compile
        if (!pp.fut.valid()) continue;
        if (pp.fut.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return; // still compiling — try next tick
    }

    // Stale build? A newer set_graph() superseded us.
    if (pending_generation_ != compile_generation_) {
        for (auto& pp : pending_passes_) if (pp.fut.valid()) pp.fut.get();
        pending_passes_.clear();
        pending_active_     = false;
        pending_generation_ = 0;
        log_info("discarded stale PassPlan compile result");
        return;
    }

    // Collect results, build new pipelines.
    std::vector<PassExec> new_passes;
    new_passes.reserve(pending_passes_.size());

    for (auto& pp : pending_passes_) {
        PassExec pe;
        pe.node_id          = pp.node_id;
        pe.output_resources = pp.output_resources;
        pe.input_resources  = std::move(pp.input_resources);
        pe.output_layout_is_general = false;
        pe.bypassed         = pp.bypassed;

        if (pp.kind == PassKind::Dispatch) {
            if (pp.bypassed) {
                // Bypassed: no shader, no pipeline; executor clears to zero.
            } else {
                CompileResult r = pp.fut.get();
                if (!r.success) {
                    set_error_(EngineErrorCode::ShaderCompile,
                               "Node " + pp.name + ": " + r.error_log,
                               EnginePhase::GraphCompileFinish, pp.node_id);
                    pending_passes_.clear();
                    pending_active_     = false;
                    pending_generation_ = 0;
                    return;
                }
                cache_->store(pp.variant_key, r.spirv);
                if (!create_pass_pipeline_(pe, pp.node_id, pp.type_id,
                        "pipeline creation failed for " + pp.name,
                        pp.variant_key, r.spirv)) {
                    pending_passes_.clear();
                    pending_active_     = false;
                    pending_generation_ = 0;
                    return;
                }
            }
            assign_bindless_slots_(pe);
        }
        new_passes.push_back(std::move(pe));
    }

    // Hot-swap.
    retire_all_passes_();
    passes_                = std::move(new_passes);
    final_output_resource_ = pending_final_output_;
    installed_generation_  = pending_generation_;
    clear_error();

    // Stage 6: build ChainExec from the just-installed plan.
    populate_chains_(pending_pass_plan_);
    pending_passes_.clear();
    pending_active_     = false;
    pending_generation_ = 0;
    log_info("PassPlan installed (compiled), passes=" + std::to_string(passes_.size())
             + " generation=" + std::to_string(installed_generation_));
}


void Engine::retire_all_passes_() {
    for (auto& pe : passes_) {
        retired_passes_.push_back({
            std::move(pe.pipeline),
            MAX_FRAMES_IN_FLIGHT + 2
        });
    }
    passes_.clear();
}


// Stage 6: build ChainExec array, chain_id_of_pass_, and membership table from PassPlan. Uses precomputed chain_index_of_pass from GraphCompiler. Computes chain_in_sampled_slots to fix the mid-chain external input bug.
void Engine::populate_chains_(const PassPlan& plan) {
    chain_execs_.clear();
    chain_execs_.reserve(plan.chains.size());
    chain_id_of_pass_ = plan.chain_index_of_pass;
    chain_id_of_pass_.resize(passes_.size(), UINT32_MAX);

    // node_id -> pass index (for head/tail lookup)
    std::unordered_map<NodeId, uint32_t> pass_idx_by_node;
    pass_idx_by_node.reserve(plan.passes.size());
    for (uint32_t i = 0; i < (uint32_t)plan.passes.size(); ++i)
        pass_idx_by_node[plan.passes[i].node_id] = i;

    std::unordered_map<ChainId, std::vector<NodeId>> membership;
    membership.reserve(plan.chains.size());

    for (size_t ci = 0; ci < plan.chains.size(); ++ci) {
        const auto& ch = plan.chains[ci];
        ChainExec ce;
        ce.bypassed = ch.bypassed;

        std::vector<uint32_t> member_pis;
        for (NodeId n : ch.nodes) {
            auto it = pass_idx_by_node.find(n);
            if (it != pass_idx_by_node.end()) member_pis.push_back(it->second);
        }
        ce.member_pass_indices = member_pis;
        if (!member_pis.empty()) {
            ce.head_pass_index = member_pis.front();
            ce.tail_pass_index = member_pis.back();
        }

        // Compute chain_in_sampled_slots: walk member passes in order,
        // matching ChainShaderEmitter::build_emit_plan external input layout.
        uint32_t ext_idx = 0;
        for (size_t mi = 0; mi < member_pis.size() && ext_idx < MAX_PASS_INPUTS; ++mi) {
            const auto& pe = passes_[member_pis[mi]];
            // Need type info to know declared input count.
            const auto* vn = current_ir_.find(ch.nodes[mi]);
            const auto* type = vn ? node_lib_.find(vn->type_id) : nullptr;
            if (!type) continue;
            const uint32_t inputs_n = (uint32_t)type->inputs.size();
            for (uint32_t s = 0; s < inputs_n; ++s) {
                if (type->inputs[s].type != SocketType::Vec4) continue;
                bool is_external = true;
                if (mi > 0 && s == 0) {
                    // Socket 0 of mid-chain: check if fed by predecessor.
                    // Use the GraphIR connection to determine.
                    ResourceUUID pred_rid{ch.nodes[mi - 1], 0};
                    const auto& pass = plan.passes[member_pis[mi]];
                    for (uint32_t ri = 0; ri < (uint32_t)pass.input_resources.size(); ++ri) {
                        if (pass.input_resources[ri] == pred_rid) {
                            is_external = false; // Local — not in in_sampled_slots
                            break;
                        }
                    }
                }
                if (is_external) {
                    ce.chain_in_sampled_slots[ext_idx++] = pe.in_sampled_slots[s];
                }
            }
        }
        while (ext_idx < MAX_PASS_INPUTS)
            ce.chain_in_sampled_slots[ext_idx++] = dummy_sampled_slot_;

        // Pipeline creation (unchanged).
        if (!ch.glsl.empty() && !ch.bypassed) {
            std::optional<std::vector<uint32_t>> blob;
            if (cache_) blob = cache_->load(ch.variant_key);
            if (!blob) {
                CompileResult r = compiler_.compile_compute_sync(
                    ch.glsl, "chain_" + std::to_string(ci));
                if (r.success) {
                    if (cache_) cache_->store(ch.variant_key, r.spirv);
                    blob = std::move(r.spirv);
                } else {
                    log_warn("chain compile failed: " + r.error_log);
                }
            }
            if (blob) {
                auto pipe = std::make_unique<ComputePipeline>();
                const std::string pipe_name = "pipe_chain_" + std::to_string(ci);
                pipe->set_name(pipe_name);
                if (pipe->create(ctx_, *blob, bindless_.pipeline_layout(), nullptr)) {
                    ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                                        (uint64_t)pipe->pipeline(), pipe->name());
                    ce.pipeline = std::move(pipe);
                } else {
                    log_warn("chain pipeline creation failed: chain " + std::to_string(ci));
                }
            }
        }
        chain_execs_.push_back(std::move(ce));
        membership[(ChainId)ci] = ch.nodes;
    }

    dirty_set_.set_chain_membership(std::move(membership));
}


void Engine::tick_retired() {
    for (auto& r : retired_images_) if (r.frames_remaining > 0) --r.frames_remaining;
    retired_images_.erase(
        std::remove_if(retired_images_.begin(), retired_images_.end(),
            [&](RetiredImage& r) {
                if (r.frames_remaining == 0) {
                    if (r.img) r.img->destroy(ctx_);
                    return true;
                }
                return false;
            }),
        retired_images_.end());

	for (auto& r : retired_passes_) if (r.frames_remaining > 0) --r.frames_remaining;
	retired_passes_.erase(
		std::remove_if(retired_passes_.begin(), retired_passes_.end(),
			[&](RetiredPass& r) {
				if (r.frames_remaining == 0) {
					if (r.pipeline) r.pipeline->destroy(ctx_);
					return true;
				}
 				return false;
 			}),
 		retired_passes_.end());
}


// Stage 6: dispatch a single chain. Issues vkCmdDispatch + layout transitions for head external inputs and tail output. Bypassed chains clear to zero.
void Engine::record_chain_dispatch_(VkCommandBuffer cmd, const PushConstants& pc,
                                    uint32_t gx, uint32_t gy, size_t chain_idx) {
    if (chain_idx >= chain_execs_.size()) return;
    const ChainExec& ce = chain_execs_[chain_idx];
    if (ce.member_pass_indices.empty()) return;
    const PassExec& head = passes_[ce.head_pass_index];
    const PassExec& tail = passes_[ce.tail_pass_index];

    // Bypassed chain: clear every member's output to zero.
    if (ce.bypassed) {
        VkClearColorValue zero{};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (uint32_t pi : ce.member_pass_indices) {
            const PassExec& pe = passes_[pi];
            for (uint32_t o = 0; o < pe.output_count; ++o) {
                auto* r = resources_.get(pe.output_resources[o]);
                if (r) vkCmdClearColorImage(cmd, r->image,
                                            VK_IMAGE_LAYOUT_GENERAL,
                                            &zero, 1, &range);
            }
        }
        return;
    }
    if (!ce.pipeline) return;   // compile failed; skip silently

    // Transition external inputs to SHADER_READ_ONLY_OPTIMAL.
    for (const auto& inp : head.input_resources) {
        if (inp.node_id == 0) continue;
        auto* src = resources_.get(inp);
        if (!src) continue;
        if (src->layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
            !dirty_set_.is_dirty(inp.node_id)) continue;
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                        | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        b.oldLayout = src->layout;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.image     = src->image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
        src->layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    // Transition tail's output to GENERAL.
    for (uint32_t o = 0; o < tail.output_count; ++o) {
        auto* r = resources_.get(tail.output_resources[o]);
        if (!r) continue;
        if (r->layout != VK_IMAGE_LAYOUT_GENERAL) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            b.srcAccessMask = 0;
            b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                            | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            b.oldLayout     = r->alias_group_id > 0 ? VK_IMAGE_LAYOUT_UNDEFINED : r->layout;
            b.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            b.image         = r->image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers    = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
            r->layout = VK_IMAGE_LAYOUT_GENERAL;
            if (r->alias_group_id > 0)
                resources_.record_alias_write(r->alias_group_id, *r);
        }
    }

    // Bind + dispatch.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ce.pipeline->pipeline());
    VkDescriptorSet set = bindless_.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);

    PassPushConstants ppc{};
    ppc.global           = pc;
    // Spec: chain's param_base_slot is the head's; param_ring_idx is shared.
    ppc.param_base_slot  = (uint32_t)head.param_base_slot;
    ppc.input_count      = head.input_count;
    for (uint32_t k = 0; k < MAX_PASS_INPUTS; ++k)
        ppc.in_sampled_slots[k] = ce.chain_in_sampled_slots[k];
    // Spec fix: out_storage_slots[0] comes from the TAIL (who writes the
    // final imageStore), not the head.
    for (uint32_t t = 0; t < MAX_PASS_OUTPUTS; ++t)
        ppc.out_storage_slots[t] = tail.out_storage_slots[t];
    ppc.param_ring_idx = param_write_idx_;

    vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PassPushConstants), &ppc);
    vkCmdDispatch(cmd, gx, gy, 1);
    ++last_dispatch_count_;
}


void Engine::record_dispatch(VkCommandBuffer cmd, const PushConstants& pc) {

    last_dispatch_count_ = 0;
    if (passes_.empty()) return;
    // Ring rotation: if params were touched since last submit, advance to the
    // next slot and snapshot. NO descriptor rebind — shader picks up via
    // pc.param_ring_idx.
    if (param_dirty_) {
        const uint32_t next = (param_write_idx_ + 1) % PARAM_RING;
        std::memcpy(param_mapped_[next], param_mapped_[param_write_idx_], MAX_NODE_PARAMS * sizeof(float));
        vmaFlushAllocation(ctx_.allocator(), param_alloc_[next], 0, VK_WHOLE_SIZE);
        param_write_idx_ = next;
        param_dirty_ = false;
    }

    const uint32_t gx = (pc.resolution_x + 7) / 8;
    const uint32_t gy = (pc.resolution_y + 7) / 8;

    // Propagate dirty set through downstream graph before recording.
    dirty_set_.propagate(downstream_adj_);

    // Defensive: skip if nothing is dirty (submit() should have caught this)
    if (!dirty_set_.any()) return;

    // Stage 6: resolve aliased memory staleness before any barriers.
    // If a clean aliased resource has stale memory from another group member,
    // mark its producer dirty so the per-pass loop re-writes it.
    resolve_aliased_staleness_();

    auto transition = [&](VkImage img,
                          VkImageLayout from, VkImageLayout to,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask  = src_stage;
        b.srcAccessMask = src_access;
        b.dstStageMask  = dst_stage;
        b.dstAccessMask = dst_access;
        b.oldLayout     = from;
        b.newLayout     = to;
        b.image         = img;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    // 1. Transition dirty node images to GENERAL. Clean passes' images stay in current layout.
    for (auto& pe : passes_) {
        if (!dirty_set_.is_dirty(pe.node_id)) continue;
        for (uint32_t i = 0; i < pe.output_count; ++i) {
            const ResourceUUID& rid = pe.output_resources[i];
            auto* r = resources_.get(rid);
            if (!r) continue;
            if (r->layout != VK_IMAGE_LAYOUT_GENERAL) {
                transition(r->image,
                            r->alias_group_id > 0 ? VK_IMAGE_LAYOUT_UNDEFINED : r->layout,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                            VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
                r->layout = VK_IMAGE_LAYOUT_GENERAL;
            }
            if (r->alias_group_id > 0)
                resources_.record_alias_write(r->alias_group_id, *r);
        }
        pe.output_layout_is_general = true;
    }

    // 2. Dummy image — only transition once across its lifetime.
    if (dummy_layout_ != VK_IMAGE_LAYOUT_GENERAL) {
        transition(dummy_.image(),
                   dummy_layout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        dummy_layout_ = VK_IMAGE_LAYOUT_GENERAL;
    }

    // P1: Queue-family ownership acquire for freshly-uploaded images. Symmetric with release barrier on transfer queue. Must use IGNORED srcStageMask (acquire side).
    if (!images_needing_acquire_.empty()) {
        std::vector<VkImageMemoryBarrier2> acquires;
        acquires.reserve(images_needing_acquire_.size());
        for (auto& kv : images_needing_acquire_) {
            auto rit = image_registry_.find(kv.first);
            if (rit == image_registry_.end() || !rit->second) continue;
            VkImageMemoryBarrier2 acq{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            acq.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
            acq.srcAccessMask = 0;
            acq.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            acq.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                              | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            acq.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;  // matches release newLayout
            acq.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
            acq.srcQueueFamilyIndex = ctx_.transfer_family();
            acq.dstQueueFamilyIndex = ctx_.graphics_family();
            acq.image         = rit->second->image();
            acq.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            acquires.push_back(acq);
        }
        if (!acquires.empty()) {
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = (uint32_t)acquires.size();
            dep.pImageMemoryBarriers    = acquires.data();
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        images_needing_acquire_.clear();
    }

    // 3. output_ → GENERAL only on first use.
    if (output_layout_ != VK_IMAGE_LAYOUT_GENERAL) {
        transition(output_storage_->image(),
                   output_layout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        output_layout_ = VK_IMAGE_LAYOUT_GENERAL;
    }

    // 4. Stage 6: chains first (one dispatch per chain), then non-chain passes. Chain members skipped in per-pass loop via chain_id_of_pass_.
    bool final_pass_was_dirty = false;
#ifndef TE_FORCE_NO_FUSION
    for (size_t ci = 0; ci < chain_execs_.size(); ++ci) {
        if (!dirty_set_.is_chain_dirty((ChainId)ci)) continue;
        const size_t tail_i = chain_execs_[ci].tail_pass_index;
        if (tail_i < passes_.size()) {
            const auto& tail_pe = passes_[tail_i];
            if (std::find(tail_pe.output_resources.begin(),
                          tail_pe.output_resources.end(),
                          final_output_resource_) != tail_pe.output_resources.end()) {
                final_pass_was_dirty = true;
            }
        }
        record_chain_dispatch_(cmd, pc, gx, gy, ci);
    }
#endif

    // Incremental dispatch -- only dirty passes
    for (size_t i = 0; i < passes_.size(); ++i) {
        auto& pe = passes_[i];
#ifndef TE_FORCE_NO_FUSION
        // Chain members are dispatched by the chain walk above.
        if (i < chain_id_of_pass_.size() &&
            chain_id_of_pass_[i] != UINT32_MAX) continue;
#endif
        if (!dirty_set_.is_dirty(pe.node_id)) continue;

        // Phase 1c: bypassed passes don't read inputs — skip input barriers
        // and skip the pipeline bind. The output state is updated below as
        // usual; the clear-to-zero is issued in the dispatch branch.
        if (!pe.bypassed) {
        // Emit input barriers only for inputs whose source pass is dirty this frame or whose layout isn't already GENERAL.
            for (const auto& inp : pe.input_resources) {
                if (inp.node_id == 0) continue;  // unconnected → bound to dummy
                auto* src = resources_.get(inp);
                if (!src) continue;
                // If upstream didn't run this frame and is already in GENERAL,
                // no barrier needed — it's stable from last frame.
                if (src->layout == VK_IMAGE_LAYOUT_GENERAL &&
                    !dirty_set_.is_dirty(inp.node_id)) {
                    continue;
                }
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.image     = src->image;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dep.imageMemoryBarrierCount = 1;
                dep.pImageMemoryBarriers    = &b;
                vkCmdPipelineBarrier2(cmd, &dep);
            }
        }
        // Emit output
        for (uint32_t o = 0; o < pe.output_count; ++o) {
            const ResourceUUID& rid = pe.output_resources[o];
            auto* out_res = resources_.get(rid);
            if (out_res) {
                out_res->is_dirty = false;
                out_res->layout = VK_IMAGE_LAYOUT_GENERAL;
                if (out_res->alias_group_id > 0)
                    resources_.record_alias_write(out_res->alias_group_id, *out_res);
            }
        }

        // Phase 1c: bypassed pass -- clear each output to zero via vkCmdClearColorImage. Image stays in GENERAL. No pipeline, no dispatch.
        if (pe.bypassed) {
            VkClearColorValue zero{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            for (uint32_t o = 0; o < pe.output_count; ++o) {
                auto* out_res = resources_.get(pe.output_resources[o]);
                if (!out_res) continue;
                vkCmdClearColorImage(cmd, out_res->image,
                                     VK_IMAGE_LAYOUT_GENERAL,
                                     &zero, 1, &range);
            }
        } else if (pe.pipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pe.pipeline->pipeline());
            VkDescriptorSet set = bindless_.set();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);

            PassPushConstants ppc{};
            ppc.global           = pc;
            ppc.param_base_slot  = (uint32_t)pe.param_base_slot;
            ppc.input_count      = pe.input_count;
            for (uint32_t k = 0; k < MAX_PASS_INPUTS; ++k)
                ppc.in_sampled_slots[k] = pe.in_sampled_slots[k];
            for (uint32_t t = 0; t < MAX_PASS_OUTPUTS; ++t)
                ppc.out_storage_slots[t] = pe.out_storage_slots[t];
            ppc.param_ring_idx = param_write_idx_;

            vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PassPushConstants), &ppc);
            vkCmdDispatch(cmd, gx, gy, 1);
            ++last_dispatch_count_;
        }

        pe.output_layout_is_general = true;

        if (std::find(pe.output_resources.begin(), pe.output_resources.end(), final_output_resource_) != pe.output_resources.end()) {
            final_pass_was_dirty = true;
        }
    }

    // 5. Blit final pass output -> output_ (only if the output pass ran)
    auto* final_res = resources_.get(final_output_resource_);
    if (final_res && final_pass_was_dirty) {
        transition(final_res->image,
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT);
        final_res->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        transition(output_storage_->image(),
                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[0]  = {0, 0, 0};
        blit.srcOffsets[1]  = {static_cast<int32_t>(pc.resolution_x), static_cast<int32_t>(pc.resolution_y), 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[0]  = {0, 0, 0};
        blit.dstOffsets[1]  = {static_cast<int32_t>(output_w_), static_cast<int32_t>(output_h_), 1};

        vkCmdBlitImage(cmd,
                       final_res->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       output_storage_->image(),  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit,
                       VK_FILTER_LINEAR);

        transition(output_storage_->image(),
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        output_layout_ = VK_IMAGE_LAYOUT_GENERAL;

        transition(final_res->image,
                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        final_res->layout = VK_IMAGE_LAYOUT_GENERAL;
    }
}


// Parameter SSBO upload
void Engine::mark_node_dirty(NodeId node_id) {
    dirty_set_.mark_node(node_id);
    any_pass_dirty_.store(true);
    mark_downstream_dirty_(node_id);
}


uint64_t Engine::republish_last_frame(uint64_t generation) {
    if (last_presented_pixels_.empty()) return 0;
    return async_.publish_synthetic(last_presented_pixels_,
                                     last_presented_w_,
                                     last_presented_h_,
                                     generation);
}


void Engine::stash_last_presented(const std::vector<float>& pixels,
                                  uint32_t w, uint32_t h, uint64_t /*generation*/) {
    last_presented_pixels_ = pixels;
    last_presented_w_ = w;
    last_presented_h_ = h;
}


void Engine::mark_downstream_dirty_(NodeId root) {
    any_pass_dirty_.store(true);
    std::unordered_set<NodeId> dirty;
    std::vector<NodeId> q;
    q.push_back(root);
    dirty.insert(root);
    size_t head = 0;
    while (head < q.size()) {
        NodeId cur = q[head++];
        auto it = downstream_adj_.find(cur);
        if (it == downstream_adj_.end()) continue;
        for (NodeId nxt : it->second) {
            if (dirty.insert(nxt).second) q.push_back(nxt);
        }
    }
    for (NodeId nid : dirty) {
        if (auto* r = resources_.get({nid, 0})) r->is_dirty = true;
    }
}


void Engine::update_node_params_by_id(NodeId node_id, const std::vector<float>& params) {
    TE_GUARD_READY(;);

    if (param_write_idx_ >= PARAM_RING) return;
    void* mapped = param_mapped_[param_write_idx_];
    if (!mapped) return;
    auto it = param_base_slot_.find(node_id);
    if (it == param_base_slot_.end()) {
        log_warn("update_node_params_by_id: unknown node id " + std::to_string(node_id));
        return;
    }
    const int base = it->second;
    if (base < 0) return;
    const size_t cap = (base >= 0 && (size_t)base < MAX_NODE_PARAMS)
                         ? (MAX_NODE_PARAMS - (size_t)base) : 0;
    if (params.size() > cap) {
        log_warn("update_node_params_by_id: truncating node " + std::to_string(node_id));
    }
    const size_t n = std::min(params.size(), cap);
    if (n > 0) {
        std::memcpy(static_cast<float*>(mapped) + base, params.data(), n * sizeof(float));
        vmaFlushAllocation(ctx_.allocator(), param_alloc_[param_write_idx_],
                           (VkDeviceSize)(base * sizeof(float)),
                           (VkDeviceSize)(n    * sizeof(float)));
    }
    param_dirty_ = true;
    // Seed dirty_set_ (Layer 2) before mark_downstream_dirty_ (Layer 3). Without this, record_dispatch would early-exit and re-publish stale pixels. See DirtySet.hpp for three-layer state machine.
    dirty_set_.mark_node(node_id);
    mark_downstream_dirty_(node_id);
}


void Engine::update_node_params_by_name(NodeId node_id,
                                        const std::unordered_map<std::string, float>& kv) {
    TE_GUARD_READY(;);

    if (!param_mapped_[param_write_idx_]) return;

    auto it = param_base_slot_.find(node_id);
    if (it == param_base_slot_.end()) {
        log_warn("update_node_params_by_name: unknown node id "
                 + std::to_string(node_id));
        return;
    }
    const int base = it->second;

    const auto* vn = current_ir_.find(node_id);
    if (!vn) {
        log_warn("update_node_params_by_name: node " + std::to_string(node_id)
               + " not in current IR");
        return;
    }
    const auto* type = node_lib_.find(vn->type_id);
    if (!type) {
        log_warn("update_node_params_by_name: unknown type '" + vn->type_id
               + "' for node " + std::to_string(node_id));
        return;
    }

    auto* dst = static_cast<float*>(param_mapped_[param_write_idx_]);

    // Phase 6: warn on keys the manifest doesn't know about -- contract violations between Python and JSON.
    std::unordered_set<std::string> known;
    known.reserve(type->params.size());
    for (auto& p : type->params) known.insert(p.name);
    for (auto& [k, _v] : kv) {
        if (!known.count(k)) {
            log_warn("update_node_params_by_name: node " + std::to_string(node_id)
                   + " (type '" + type->id + "') has no param named '" + k
                   + "' -- ignoring.");
        }
    }

    int min_slot = INT_MAX, max_slot = -1;
    for (size_t i = 0; i < type->params.size(); ++i) {
        auto found = kv.find(type->params[i].name);
        if (found == kv.end()) continue;
        const int slot = base + static_cast<int>(i);
        if (slot < 0 || slot >= static_cast<int>(MAX_NODE_PARAMS)) {
            log_warn("update_node_params_by_name: slot " + std::to_string(slot)
                   + " out of range for param '" + type->params[i].name
                   + "' on node " + std::to_string(node_id)
                   + " -- DROPPED (bug: GraphCompiler should have rejected this graph).");
            continue;
        }
        dst[slot] = found->second;
        if (slot < min_slot) min_slot = slot;
        if (slot > max_slot) max_slot = slot;
    }
    if (max_slot >= 0) {
        const VkDeviceSize off  = (VkDeviceSize)(min_slot * sizeof(float));
        const VkDeviceSize size = (VkDeviceSize)((max_slot - min_slot + 1) * sizeof(float));
        vmaFlushAllocation(ctx_.allocator(), param_alloc_[param_write_idx_], off, size);
    }
    param_dirty_ = true;
    // See update_node_params_by_id above. Seed Layer 2 (DirtySet) ourselves or next record_dispatch will early-exit.
    dirty_set_.mark_node(node_id);
    mark_downstream_dirty_(node_id);
}


bool Engine::upload_image(uint64_t node_id, const float* pixels,
                          uint32_t width, uint32_t height) {
    TE_GUARD_READY(false);

    // Hand off any existing image as a recycled buffer if shapes match.
    std::unique_ptr<Image> recycled;
    auto it = image_registry_.find(node_id);
    if (it != image_registry_.end()) {
        recycled = std::move(it->second);
        image_registry_.erase(it);
    }

    const uint64_t ticket = uploader_.submit(ctx_, node_id, pixels, width, height,
                                             std::move(recycled));
    if (ticket == 0) {
        // Ring full -- Python should retry next tick. Surface a soft failure.
        set_error_(EngineErrorCode::ImageUploadRingFull,
                   "uploader ring full for node " + std::to_string(node_id),
                   EnginePhase::ImageUpload, node_id);
        return false;
    }
    pending_uploads_.push_back({node_id, ticket});
    return true;
}


bool Engine::release_image(uint64_t node_id) {
    TE_GUARD_READY(false);

    auto it = image_registry_.find(node_id);
    if (it == image_registry_.end()) {
        set_error_(EngineErrorCode::ImageReleaseUnknown,
                   "no image registered for node " + std::to_string(node_id),
                   EnginePhase::ImageRelease, node_id);
        return false;
    }

    bool has_pending = false;
    for (auto& p : pending_uploads_) if (p.node_id == node_id) { has_pending = true; break; }
    if (has_pending) uploader_.drain(ctx_);

    if (it->second) it->second->destroy(ctx_);
    image_registry_.erase(it);
    images_needing_acquire_.erase(node_id);

    auto sit = ext_sampled_slot_.find(node_id);
    if (sit != ext_sampled_slot_.end()) {
        bindless_.free_sampled_slot(sit->second);
        ext_sampled_slot_.erase(sit);
    }
    // Patch passes that referenced this external image → fall back to dummy.
    for (auto& pe : passes_) {
        if (pe.node_id != node_id) continue;
        for (uint32_t i = 0; i < pe.input_count && i < MAX_PASS_INPUTS; ++i)
            if (pe.input_resources[i].node_id == 0)
                pe.in_sampled_slots[i] = dummy_sampled_slot_;
    }
    return true;
}


void Engine::set_error_(EngineErrorCode code,
                        std::string message,
                        EnginePhase phase,
                        NodeId failed_node,
                        uint64_t graph_generation) {
    last_error_record_.code             = code;
    last_error_record_.message          = std::move(message);
    last_error_record_.failed_node      = failed_node;
    last_error_record_.graph_generation = graph_generation;
    last_error_record_.phase            = phase;

    // Mirror to legacy fields for callers that still read the old API.
    last_error_     = last_error_record_.message;
    failed_node_id_ = failed_node;

    // Log a human-readable prefix including the code + phase.
    log_error("[" + std::string(engine_error_code_name(code)) + " @ "
              + std::string(engine_phase_name(phase)) + "] "
              + last_error_record_.message
              + (failed_node ? " (node " + std::to_string(failed_node) + ")" : ""));
}


} // namespace te
