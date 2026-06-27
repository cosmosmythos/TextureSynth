#include "engine/Engine.hpp"
#include "engine/EngineBarriers.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <climits>
#include <thread>
#include <unordered_set>

namespace te {

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
    if (!ensure_dummy_images_()) {
        set_error_(EngineErrorCode::InitFailed,
                   "dummy images creation failed",
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

    output_storage_slot_ = bindless_.alloc_storage_slot();
    if (output_storage_slot_ == BindlessTable::INVALID_SLOT) {
        set_error_(EngineErrorCode::InitFailed,
                   "output storage bindless slot allocation failed",
                   EnginePhase::Init);
        goto rollback;
    }
    bindless_.write_storage(ctx_, output_storage_slot_, output_storage_->view());

    // Bind the single dummy image to bindless sampled slot 0.
    dummy_slot_ = 0;
    bindless_.write_sampled(ctx_, dummy_slot_,
                            dummy_image_.view(), VK_IMAGE_LAYOUT_GENERAL);

    // Stage 8: per-pass GPU timestamp pools.
    if (!create_timestamp_pools_()) {
        set_error_(EngineErrorCode::InitFailed,
                   "timestamp pool creation failed",
                   EnginePhase::Init);
        goto rollback;
    }

    if (!create_final_copy_pipeline_()) {
        set_error_(EngineErrorCode::InitFailed,
                   "final copy pipeline creation failed",
                   EnginePhase::Init);
        goto rollback;
    }

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


bool Engine::ensure_dummy_images_() {
    // Single RGBA32F 1x1 dummy at value (0,0,0,1). texelFetch on a sampled
    // image view performs automatic format conversion per the Vulkan spec
    // (Data Format Conversion chapter), so one RGBA32F dummy safely serves
    // all channel/depth expectations.
    const VkClearColorValue clear = {0.0f, 0.0f, 0.0f, 1.0f};

    if (!dummy_image_.create(ctx_, 1, 1)) {
        log_error("dummy image create failed");
        return true; // non-fatal; image will contain garbage
    }

    VkCommandPool tmp_pool = VK_NULL_HANDLE;
    VkCommandBuffer tmp_cmd = VK_NULL_HANDLE;
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = ctx_.graphics_family();
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    if (vkCreateCommandPool(ctx_.device(), &cpci, nullptr, &tmp_pool) != VK_SUCCESS)
        return true;
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool        = tmp_pool;
    cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx_.device(), &cbai, &tmp_cmd) != VK_SUCCESS)
        { vkDestroyCommandPool(ctx_.device(), tmp_pool, nullptr); return true; }
    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(tmp_cmd, &cbbi) != VK_SUCCESS)
        { vkFreeCommandBuffers(ctx_.device(), tmp_pool, 1, &tmp_cmd);
          vkDestroyCommandPool(ctx_.device(), tmp_pool, nullptr); return true; }

    VkImageSubresourceRange sr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    {
        VkImageMemoryBarrier2 ib{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        ib.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        ib.srcAccessMask = 0;
        ib.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        ib.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        ib.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        ib.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        ib.image         = dummy_image_.image();
        ib.subresourceRange = sr;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers    = &ib;
        vkCmdPipelineBarrier2(tmp_cmd, &di);

        vkCmdClearColorImage(tmp_cmd, dummy_image_.image(),
                             VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &sr);
    }

    if (vkEndCommandBuffer(tmp_cmd) != VK_SUCCESS)
        { vkFreeCommandBuffers(ctx_.device(), tmp_pool, 1, &tmp_cmd);
          vkDestroyCommandPool(ctx_.device(), tmp_pool, nullptr); return true; }
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &tmp_cmd;
    {
        std::lock_guard<std::mutex> lk(ctx_.graphics_queue_mutex());
        if (vkQueueSubmit(ctx_.graphics_queue(), 1, &si, VK_NULL_HANDLE) != VK_SUCCESS)
            { vkFreeCommandBuffers(ctx_.device(), tmp_pool, 1, &tmp_cmd);
              vkDestroyCommandPool(ctx_.device(), tmp_pool, nullptr); return true; }
    }
    vkQueueWaitIdle(ctx_.graphics_queue());
    vkFreeCommandBuffers(ctx_.device(), tmp_pool, 1, &tmp_cmd);
    vkDestroyCommandPool(ctx_.device(), tmp_pool, nullptr);

    log_info("dummy image created (1x1 RGBA32F)");
    return true;
}

bool Engine::create_final_copy_pipeline_() {
    static constexpr const char* kFinalCopyGlsl = R"GLSL(
#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform texture2D u_sampled[];
layout(set = 0, binding = 1, rgba32f) writeonly uniform image2D u_storage[];

layout(push_constant, std430) uniform PC {
    uint  resolution_x;
    uint  resolution_y;
    uint  seed;
    float time;
    uint  out_storage_slots[4];
    uint  param_base_slot;
    uint  input_count;
    uint  param_ring_idx;
    uint  in_sampled_slots[8];
    uint  pass_index;
} pc;

void main() {
    ivec2 dst = ivec2(gl_GlobalInvocationID.xy);
    ivec2 dst_size = ivec2(int(pc.resolution_x), int(pc.resolution_y));
    if (dst.x >= dst_size.x || dst.y >= dst_size.y) return;

    uint src_slot = pc.in_sampled_slots[0];
    ivec2 src_size = textureSize(u_sampled[nonuniformEXT(src_slot)], 0);
    ivec2 src_max = max(src_size - ivec2(1), ivec2(0));
    ivec2 src = min((dst * src_size) / max(dst_size, ivec2(1)), src_max);

    vec4 value = texelFetch(u_sampled[nonuniformEXT(src_slot)], src, 0);
    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], dst, value);
}
)GLSL";

    CompileResult r = compiler_.compile_compute_sync(kFinalCopyGlsl, "final_copy");
    if (!r.success) {
        log_error("final_copy shader compile failed: " + r.error_log);
        return false;
    }

    auto pipeline = std::make_unique<ComputePipeline>();
    pipeline->set_name("pipe_final_copy");
    if (!pipeline->create(ctx_, r.spirv, bindless_.pipeline_layout(), nullptr)) {
        log_error("final_copy pipeline creation failed");
        return false;
    }
    ctx_.set_debug_name(VK_OBJECT_TYPE_PIPELINE,
                        (uint64_t)pipeline->pipeline(),
                        pipeline->name());
    final_copy_pipeline_ = std::move(pipeline);
    return true;
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
        for (auto& sp : ce.sub_pipelines) {
            if (sp) sp->destroy(ctx_);
        }
        for (auto& intermedi : ce.intermediates) {
            if (intermedi.image) intermedi.image->destroy(ctx_);
        }
    }
    chain_execs_.clear();
    chain_id_of_pass_.clear();

    if (final_copy_pipeline_) {
        final_copy_pipeline_->destroy(ctx_);
        final_copy_pipeline_.reset();
    }

    dummy_slot_ = BindlessTable::INVALID_SLOT;
    output_storage_slot_ = BindlessTable::INVALID_SLOT;
    output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    dummy_layout_  = VK_IMAGE_LAYOUT_UNDEFINED;
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

    dummy_image_.destroy(ctx_);
    uploader_.shutdown(ctx_);
    async_.shutdown(ctx_);
    for (auto& r : retired_images_) if (r.img) r.img->destroy(ctx_);
    retired_images_.clear();
    destroy_timestamp_pools_();
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


uint64_t Engine::set_graph(const Graph& graph) {
    TE_GUARD_READY(0);
    clear_error();

    // Topology change must drain in-flight async work before we tear down
    // descriptor sets, pipelines, or per-node images.
    async_.drain(ctx_);

    auto ir_result = validate_graph(graph, node_lib_);
    if (!ir_result.success) {
        log_warn("Graph validation: " + ir_result.error);
        return 0;
    }

    // SD-style depth inheritance: stamp the graph default (from sidebar)
    // and resolve each node's depth before fusion compiles.
    ir_result.ir.graph_default_depth = graph_default_depth_;
    resolve_node_depths(ir_result.ir);

    auto compile_result = FusedGraphCompiler::compile(ir_result.ir, node_lib_, ir_result.ir.output_node);
    if (!compile_result.success) {
        set_error_(EngineErrorCode::GraphCompile, compile_result.error,
                   EnginePhase::GraphSubmit);
        return 0;
    }

    // Dump chain debug info (GLSL + structure) before the PassPlan is consumed.
    chain_debug_.clear();
    for (uint32_t ci = 0; ci < (uint32_t)compile_result.pass_plan.chains.size(); ++ci) {
        const auto& ch = compile_result.pass_plan.chains[ci];
        ChainDebugInfo di;
        di.chain_index        = ci;
        di.nodes              = ch.nodes;
        di.sub_pass_count     = ch.sub_pass_count;
        di.intermediate_count = ch.intermediate_count;
        di.colors_used        = ch.coloring.colors_used;
        di.spilled_count      = (uint32_t)ch.coloring.spilled.size();
        di.shared_slots       = ch.coloring.shared_slot_count;
        di.glsl              = ch.glsl;
        di.sub_pass_glsl     = ch.sub_pass_glsl;
        di.variant_key       = ch.variant_key;
        chain_debug_.push_back(std::move(di));
    }

    current_graph_ = graph;
    current_graph_.output_node = ir_result.ir.output_node;
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
                                        &compile_result.pass_plan.color_classes,
                                        &compile_result.pass_plan.active_resources)) {
        set_error_(EngineErrorCode::GraphCompile, rerr, EnginePhase::GraphSubmit);
        return 0;
    }

    param_base_slot_     = std::move(compile_result.param_base_slot);
    total_param_floats_  = compile_result.total_param_floats;

    // Seed param SSBOs with manifest defaults (params AND float-input defaults).
    seed_param_ssbo_defaults_();

    const uint64_t gen = ++compile_generation_;

    for (auto& pp : pending_passes_) if (pp.fut.valid()) pp.fut.wait();
    pending_passes_.clear();

    const auto& cip = compile_result.pass_plan.chain_index_of_pass;

    // Step 3: Check per-node cache (skip chain-member passes — their shader lives in the chain).
    bool all_cached = true;
    std::vector<std::vector<uint32_t>> cached_spv(compile_result.pass_plan.passes.size());
    for (size_t i = 0; i < compile_result.pass_plan.passes.size(); ++i) {
        const auto& pass = compile_result.pass_plan.passes[i];
        if (pass.kind != PassKind::Compute) continue;
        if (i < cip.size() && cip[i] != UINT32_MAX) continue;  // chain member: skip
        auto blob = cache_->load(pass.variant_key);
        if (!blob) { all_cached = false; break; }
        cached_spv[i] = std::move(*blob);
    }

    // Also check fused chain cache.
    std::vector<std::vector<uint32_t>> cached_fused_spv(compile_result.pass_plan.chains.size());
    if (all_cached) {
        for (size_t ci = 0; ci < compile_result.pass_plan.chains.size(); ++ci) {
            const auto& ch = compile_result.pass_plan.chains[ci];
            if (ch.glsl.empty() || ch.bypassed) continue;
            auto blob = cache_->load(ch.variant_key);
            if (!blob) { all_cached = false; break; }
            cached_fused_spv[ci] = std::move(*blob);
        }
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
            pe.input_formats    = pass.input_formats;
            pe.param_base_slot  = pass.param_base_slot;
            pe.bypassed         = pass.bypassed;

            if (pass.kind == PassKind::Compute) {
                bool is_chain_member = (i < cip.size() && cip[i] != UINT32_MAX);
                if (!pass.bypassed && !is_chain_member) {
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

    // Step 5: Async path — set chain_member flag, skip compile for chain members.
    pending_passes_.clear();
    pending_passes_.reserve(compile_result.pass_plan.passes.size());
    for (size_t i = 0; i < compile_result.pass_plan.passes.size(); ++i) {
        const auto& pass = compile_result.pass_plan.passes[i];
        PendingPass pp;
        pp.name             = "node_" + std::to_string(pass.node_id);
        pp.type_id          = pass.type_id;
        pp.input_count      = pass.input_socket_count;
        pp.output_resources = pass.output_resources;
        pp.input_resources  = pass.input_resources;
        pp.input_formats    = pass.input_formats;
        pp.node_id          = pass.node_id;
        pp.kind             = pass.kind;
        pp.variant_key      = pass.variant_key;
        pp.bypassed         = pass.bypassed;
        pp.param_base_slot  = pass.param_base_slot;
        bool is_chain_member = (i < cip.size() && cip[i] != UINT32_MAX);
        pp.chain_member     = is_chain_member;
        if (pass.kind == PassKind::Compute && !pass.bypassed && !is_chain_member) {
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
    // Check the node exists before building a graph that would fail validation.
    bool node_exists = false;
    for (const auto& n : current_graph_.nodes) {
        if (n.id == node_id) { node_exists = true; break; }
    }
    if (!node_exists) {
        return 0;
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

    struct T { NodeId node; uint32_t socket; std::string name; };
    std::vector<T> targets;
    if (current_graph_.output_targets.empty()) {
        targets.push_back({current_graph_.output_node, current_graph_.output_socket, "output"});
    } else {
        for (const auto& t : current_graph_.output_targets) {
            targets.push_back({t.source_node, t.source_socket, t.name});
        }
    }

    const NodeId original_active = current_graph_.output_node;
    const uint32_t original_socket = current_graph_.output_socket;
    for (const auto& t : targets) {
        if (t.node != current_graph_.output_node || t.socket != current_graph_.output_socket) {
            Graph g = current_graph_;
            g.output_node = t.node;
            g.output_socket = t.socket;
            uint64_t gen = set_graph(g);
            if (gen == 0) return out;
        }
        BakedImage img = readback_sync();
        if (img.pixels.empty()) return out;
        img.name = t.name;
        out.push_back(std::move(img));
    }
    if (current_graph_.output_node != original_active ||
        current_graph_.output_socket != original_socket) {
        Graph g = current_graph_;
        g.output_node = original_active;
        g.output_socket = original_socket;
        set_graph(g);
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


// Stage 8: create timestamp query pools (one per in-flight slot).
// Uses a one-time command buffer to reset all pools (avoids needing hostQueryReset).
bool Engine::create_timestamp_pools_() {
    VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = 256;  // plenty for any reasonable graph (128 passes × 2)
    for (auto& tp : ts_pools_) {
        if (vkCreateQueryPool(ctx_.device(), &qci, nullptr, &tp.pool) != VK_SUCCESS) {
            log_error("Engine: VkQueryPool creation failed");
            return false;
        }
        tp.capacity = qci.queryCount;
    }

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    cpci.queueFamilyIndex = ctx_.graphics_family();
    VkCommandPool tmp_pool;
    vkCreateCommandPool(ctx_.device(), &cpci, nullptr, &tmp_pool);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = tmp_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer tmp_cmd;
    vkAllocateCommandBuffers(ctx_.device(), &cbai, &tmp_cmd);

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(tmp_cmd, &begin);

    for (auto& tp : ts_pools_) {
        vkCmdResetQueryPool(tmp_cmd, tp.pool, 0, tp.capacity);
    }

    vkEndCommandBuffer(tmp_cmd);

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(ctx_.device(), &fci, nullptr, &fence);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &tmp_cmd;
    {
        std::lock_guard<std::mutex> lk(ctx_.graphics_queue_mutex());
        vkQueueSubmit(ctx_.graphics_queue(), 1, &submit, fence);
    }
    vkWaitForFences(ctx_.device(), 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(ctx_.device(), fence, nullptr);
    vkDestroyCommandPool(ctx_.device(), tmp_pool, nullptr);

    return true;
}

void Engine::destroy_timestamp_pools_() {
    for (auto& tp : ts_pools_) {
        if (tp.pool) {
            vkDestroyQueryPool(ctx_.device(), tp.pool, nullptr);
            tp.pool = VK_NULL_HANDLE;
        }
        tp.capacity = 0;
        tp.cached_timings.clear();
    }
}

void Engine::poll_timestamps_() {
    // Read the most recently submitted pool (previous write index).
    const uint32_t pool_idx = ts_pool_write_idx_;
    auto& tp = ts_pools_[pool_idx];
    if (tp.pool == VK_NULL_HANDLE || tp.capacity == 0) return;

    const uint32_t query_count = tp.capacity;
    std::vector<uint64_t> raw(query_count * 2, 0);  // +availability per slot
    const VkResult r = vkGetQueryPoolResults(
        ctx_.device(), tp.pool, 0, query_count,
        raw.size() * sizeof(uint64_t), raw.data(),
        sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r != VK_SUCCESS && r != VK_NOT_READY) return;

    // Decode into cached timings.
    const double period = ctx_.timestamp_period();
    tp.cached_timings.clear();
    // Query 0 = frame start (TOP_OF_PIPE). Dispatch pairs start at query 1:
    // start query = 1+2*i, end query = 2+2*i
    for (uint32_t i = 0; i < query_count / 2; ++i) {
        const uint32_t sq = 1 + 2 * i;  // start query index
        const uint32_t eq = 2 + 2 * i;  // end query index
        if (eq >= query_count) break;
        const double start_ns = (double)raw[sq * 2] * period;
        const double end_ns   = (double)raw[eq * 2] * period;
        if (!raw[sq * 2 + 1] || !raw[eq * 2 + 1]) continue;
        if (end_ns <= start_ns) continue;
        PassTiming pt;
        pt.pass_index  = i;
        pt.duration_us = (end_ns - start_ns) / 1000.0;
        pt.available   = true;
        tp.cached_timings.push_back(std::move(pt));
    }
    // Store a flat snapshot (merged across all pools, latest-read wins).
    last_pass_timings_ = tp.cached_timings;
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
        // Check for in-flight upload — drain to wait for GPU, then clean up.
        auto pit = std::remove_if(pending_uploads_.begin(), pending_uploads_.end(),
                                  [&](const PendingUpload& p){ return p.node_id == node_id; });
        if (pit != pending_uploads_.end()) {
            pending_uploads_.erase(pit, pending_uploads_.end());
            uploader_.drain(ctx_);
            return true;
        }
        // No image registered and no pending upload — no-op.
        return true;
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
    // Patch passes that referenced this external image → fall back to per-format dummy.
    for (auto& pe : passes_) {
        if (pe.node_id != node_id) continue;
        for (uint32_t i = 0; i < pe.input_count && i < MAX_PASS_INPUTS; ++i) {
            if (pe.input_resources[i].node_id == 0) {
                pe.in_sampled_slots[i] = dummy_slot_;
            }
        }
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


std::string Engine::dump_chain_debug() const {
    std::ostringstream oss;
    oss << "=== CHAIN DEBUG (" << chain_debug_.size() << " chains) ===\n";
    for (const auto& di : chain_debug_) {
        oss << "\n--- Chain " << di.chain_index << " ---\n";
        oss << "  nodes: [";
        for (size_t i = 0; i < di.nodes.size(); ++i) {
            if (i) oss << ", ";
            oss << di.nodes[i];
        }
        oss << "]\n";
        oss << "  sub_pass_count=" << di.sub_pass_count
            << " intermediates=" << di.intermediate_count
            << " regs=" << di.colors_used
            << " spilled=" << di.spilled_count
            << " shared_slots=" << di.shared_slots << "\n";
        if (!di.glsl.empty()) {
            oss << "  --- GLSL ---\n";
            // Indent every line
            std::istringstream ls(di.glsl);
            std::string line;
            while (std::getline(ls, line)) oss << "  " << line << "\n";
        }
        for (uint32_t sp = 0; sp < (uint32_t)di.sub_pass_glsl.size(); ++sp) {
            if (!di.sub_pass_glsl[sp].empty()) {
                oss << "  --- Sub-pass " << sp << " GLSL ---\n";
                std::istringstream ls(di.sub_pass_glsl[sp]);
                std::string line;
                while (std::getline(ls, line)) oss << "  " << line << "\n";
            }
        }
    }
    return oss.str();
}


} // namespace te
