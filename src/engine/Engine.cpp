#include "engine/Engine.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeRegistryLoader.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <climits>
#include <unordered_set>

namespace te {

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
bool Engine::init(VkSurfaceKHR surface,
                  const char** extra_inst_exts, uint32_t extra_inst_ext_count,
                  bool enable_validation,
                  const std::string& cache_dir,
                  const std::string& nodes_dir,
                  const std::string& glsl_dir) {

    VulkanContextDesc d{};
    d.enable_validation = enable_validation;
    d.surface = surface;
    d.extra_instance_extensions = extra_inst_exts;
    d.extra_instance_extension_count = extra_inst_ext_count;
    if (!ctx_.init(d)) return false;

    cache_ = std::make_unique<ShaderCache>(cache_dir);

    std::string err;
    if (NodeRegistryLoader::load_from_directory(node_lib_, nodes_dir, glsl_dir, &err) == 0) {
        log_warn("no nodes loaded: " + err);
    }

    output_storage_ = std::make_unique<Image>();
    if (!output_storage_->create(ctx_, output_w_, output_h_)) return false;
    if (!async_.init(ctx_, 4096, 4096)) { log_error("AsyncReadback init failed"); return false; }
    if (!uploader_.init(ctx_)) { log_error("ImageUploader init failed"); return false; }
    if (!ensure_dummy_image_()) return false;

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
                log_error("Engine: param SSBO alloc failed");
                return false;
            }
            param_mapped_[i] = ai.pMappedData;
            std::memset(param_mapped_[i], 0, (size_t)sz);
            vmaFlushAllocation(ctx_.allocator(), param_alloc_[i], 0, VK_WHOLE_SIZE);
        }
    }

    if (!create_global_samplers_()) return false;

    // Bindless table — ONE descriptor set for the whole engine lifetime.
    if (!bindless_.init(ctx_, sampler_repeat_, sampler_clamp_, sampler_mirror_,
                        sizeof(PassPushConstants))) {
        log_error("BindlessTable init failed");
        return false;
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
    return true;
}


bool Engine::ensure_dummy_image_() {
    return dummy_.create(ctx_, 1, 1);
}


void Engine::shutdown() {
    if (ctx_.device()) vkDeviceWaitIdle(ctx_.device());
    async_.drain(ctx_);
    uploader_.drain(ctx_);

    for (auto& pp : pending_passes_) if (pp.fut.valid()) pp.fut.wait();
    pending_passes_.clear();

    for (auto& p : passes_) {
        if (p.pipeline) p.pipeline->destroy(ctx_);
        release_bindless_slots_(p);
    }
    passes_.clear();
    for (auto& r : retired_passes_) {
        if (r.pipeline) r.pipeline->destroy(ctx_);
    }
    retired_passes_.clear();

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
    if (vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_repeat_) != VK_SUCCESS) return false;

    // Clamp
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_clamp_) != VK_SUCCESS) return false;

    // Mirror
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    if (vkCreateSampler(ctx_.device(), &sci, nullptr, &sampler_mirror_) != VK_SUCCESS) return false;

    return true;
}


void Engine::destroy_global_samplers_() {
    if (sampler_repeat_) { vkDestroySampler(ctx_.device(), sampler_repeat_, nullptr); sampler_repeat_ = VK_NULL_HANDLE; }
    if (sampler_clamp_)  { vkDestroySampler(ctx_.device(), sampler_clamp_, nullptr);  sampler_clamp_  = VK_NULL_HANDLE; }
    if (sampler_mirror_) { vkDestroySampler(ctx_.device(), sampler_mirror_, nullptr); sampler_mirror_ = VK_NULL_HANDLE; }
}


void Engine::assign_bindless_slots_(PassExec& pe) {

    uint32_t out_slot;
    auto it_o = res_storage_slot_.find(pe.output_resource);
    if (it_o == res_storage_slot_.end()) {
        out_slot = bindless_.alloc_storage_slot();
        if (out_slot == BindlessTable::INVALID_SLOT) {
            log_error("Bindless storage exhausted at pass node " + std::to_string(pe.node_id));
            out_slot = 0;  // will write to slot 0; visible debug failure, not crash
        } else {
            res_storage_slot_[pe.output_resource] = out_slot;
            if (auto* r = resources_.get(pe.output_resource))
                bindless_.write_storage(ctx_, out_slot, r->view);
        }
    } else {
        out_slot = it_o->second;
    }
    pe.out_storage_slot = out_slot;

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


void Engine::release_bindless_slots_(PassExec& pe) {
    // We don't free resource slots here — they're owned by ResourceManager
    // lifecycle (flushed when ResourceManager::retire_all runs in set_graph).
    // External image slots are freed in release_image / shutdown.
    (void)pe;
}


uint64_t Engine::set_graph(const Graph& graph) {
    last_error_.clear();
    failed_node_id_ = 0;

    // Topology change must drain in-flight async work before we tear down
    // descriptor sets, pipelines, or per-node images.
    async_.drain(ctx_);

    auto ir_result = validate_graph(graph, node_lib_);
    if (!ir_result.success) {
        last_error_ = ir_result.error;
        log_error("Graph validation failed: " + ir_result.error);
        return 0;
    }

    auto compile_result = GraphCompiler::compile(ir_result.ir, node_lib_, texture_format_);
    if (!compile_result.success) {
        last_error_ = compile_result.error;
        log_error("Graph compile failed: " + compile_result.error);
        return 0;
    }

    current_graph_ = graph;
    current_ir_    = std::move(ir_result.ir);
    current_revision_++;
    rebuild_downstream_adj_();
    any_pass_dirty_.store(true);
    dirty_set_.mark_topology_change();
    exec_generation_ = 0;
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
    if (!resources_.allocate_for_graph(ctx_, current_ir_, output_w_, output_h_,
                                       texture_format_, &rerr)) {
        last_error_ = rerr;
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
    param_write_idx_ = 0;

    const uint64_t gen = ++compile_generation_;

    for (auto& pp : pending_passes_) if (pp.fut.valid()) pp.fut.wait();
    pending_passes_.clear();

    bool all_cached = true;
    std::vector<std::vector<uint32_t>> cached_spv;
    cached_spv.reserve(compile_result.pass_plan.passes.size());
    for (auto& pass : compile_result.pass_plan.passes) {
        if (pass.kind != PassKind::Dispatch) continue;
        auto blob = cache_->load(pass.variant_key);
        if (!blob) { all_cached = false; break; }
        cached_spv.push_back(std::move(*blob));
    }

    if (all_cached) {
        passes_.clear();
        passes_.reserve(compile_result.pass_plan.passes.size());

        for (size_t i = 0; i < compile_result.pass_plan.passes.size(); ++i) {
            const auto& pass = compile_result.pass_plan.passes[i];
            PassExec pe;
            pe.node_id         = pass.node_id;
            pe.output_resource = pass.output_resource;
            pe.input_resources = pass.input_resources;
            pe.param_base_slot = pass.param_base_slot;
            pe.output_layout_is_general = false;

            if (pass.kind == PassKind::Dispatch) {
                auto pipe = std::make_unique<ComputePipeline>();
                if (!pipe->create(ctx_, cached_spv[i], bindless_.pipeline_layout())) {
                    last_error_ = "pipeline creation failed for node "
                                + std::to_string(pass.node_id);
                    return 0;
                }
                pe.pipeline = std::move(pipe);
                assign_bindless_slots_(pe);
            }
            passes_.push_back(std::move(pe));
        }
        final_output_resource_ = compile_result.pass_plan.final_output_resource;
        installed_generation_  = gen;
        log_info("PassPlan installed (cache hit), passes=" + std::to_string(passes_.size())
                 + " generation=" + std::to_string(gen));
        return gen;
    }

    pending_passes_.clear();
    pending_passes_.reserve(compile_result.pass_plan.passes.size());
    for (auto& pass : compile_result.pass_plan.passes) {
        PendingPass pp;
        pp.name            = "node_" + std::to_string(pass.node_id);
        pp.input_count     = pass.input_socket_count;
        pp.output_resource = pass.output_resource;
        pp.input_resources = pass.input_resources;
        pp.node_id         = pass.node_id;
        pp.kind            = pass.kind;
        pp.variant_key     = pass.variant_key;
        if (pass.kind == PassKind::Dispatch) {
            pp.fut = compiler_.compile_compute_async(pass.shader_glsl, pp.name);
        }
        pending_passes_.push_back(std::move(pp));
    }
    pending_active_       = true;
    pending_generation_   = gen;
    pending_final_output_ = compile_result.pass_plan.final_output_resource;
    log_info("PassPlan compile dispatched, passes=" + std::to_string(pending_passes_.size())
             + " generation=" + std::to_string(gen));
    return gen;
}


// ---------------------------------------------------------------------------
// rebuild_downstream_adj_ — build adjacency list for dirty propagation
// ---------------------------------------------------------------------------
void Engine::rebuild_downstream_adj_() {
    downstream_adj_.clear();
    for (const auto& vn : current_ir_.nodes) {
        downstream_adj_[vn.id];   // ensure key exists even for sinks
    }
    for (const auto& c : current_ir_.connections) {
        downstream_adj_[c.src_node].push_back(c.dst_node);
    }
}


void Engine::poll_pending_compiles() {

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
        pe.node_id         = pp.node_id;
        pe.output_resource = pp.output_resource;
        pe.input_resources = std::move(pp.input_resources);
        pe.output_layout_is_general = false;

        if (pp.kind == PassKind::Dispatch) {
            CompileResult r = pp.fut.get();
            if (!r.success) {
                last_error_ = "Node " + pp.name + ": " + r.error_log;
                log_error("pass compile failed for " + pp.name + ": " + r.error_log);
                failed_node_id_ = pp.node_id;
                pending_passes_.clear();
                pending_active_     = false;
                pending_generation_ = 0;
                return;
            }
            cache_->store(pp.variant_key, r.spirv);
            auto pipe = std::make_unique<ComputePipeline>();
            if (!pipe->create(ctx_, r.spirv, bindless_.pipeline_layout())) {
                last_error_ = "pipeline creation failed for " + pp.name;
                return;
            }
            pe.pipeline = std::move(pipe);
            assign_bindless_slots_(pe);
        }
        new_passes.push_back(std::move(pe));
    }

    // Hot-swap.
    retire_all_passes_();
    passes_                = std::move(new_passes);
    final_output_resource_ = pending_final_output_;
    installed_generation_  = pending_generation_;
    last_error_.clear();

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
}


void Engine::record_dispatch(VkCommandBuffer cmd, const PushConstants& pc) {

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

    // Defensive: if nothing is dirty, skip entirely (submit() should have caught this).
    if (!dirty_set_.any()) return;

    ++exec_generation_;

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

    // 1. Transition dirty node images to GENERAL.
    //    Clean passes' images stay in their current layout (GENERAL or SAMPLED).
    for (auto& pe : passes_) {
        if (!dirty_set_.is_dirty(pe.node_id)) continue;
        auto* r = resources_.get(pe.output_resource);
        if (!r) continue;
        if (r->layout != VK_IMAGE_LAYOUT_GENERAL) {
            transition(r->image,
                       r->layout, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            r->layout = VK_IMAGE_LAYOUT_GENERAL;
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

    // P1: Queue-family ownership acquire for freshly-uploaded images.
    // Symmetric with the release barrier issued on the transfer queue.
    // Must use IGNORED stage masks on srcStageMask (acquire side).
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

    // 4. Incremental dispatch — only dirty passes.
    bool final_pass_was_dirty = false;
    for (size_t i = 0; i < passes_.size(); ++i) {
        auto& pe = passes_[i];
        if (!dirty_set_.is_dirty(pe.node_id)) continue;

        auto* out_res = resources_.get(pe.output_resource);

        // Emit input barriers ONLY for inputs whose source pass is dirty this frame
        // or whose layout isn't already GENERAL (readable by compute).
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

        // Dispatch.
        if (pe.pipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pe.pipeline->pipeline());
            VkDescriptorSet set = bindless_.set();
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);

            PassPushConstants ppc{};
            ppc.global           = pc;
            ppc.out_storage_slot = pe.out_storage_slot;
            ppc.param_base_slot  = (uint32_t)pe.param_base_slot;
            ppc.input_count      = pe.input_count;
            ppc.param_ring_idx   = param_write_idx_;          
            for (uint32_t k = 0; k < MAX_PASS_INPUTS; ++k)
                ppc.in_sampled_slots[k] = pe.in_sampled_slots[k];

            vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PassPushConstants), &ppc);
            vkCmdDispatch(cmd, gx, gy, 1);
        }

        // Mark output as readable for downstream passes.
        if (out_res) {
            out_res->is_dirty = false;
            out_res->layout = VK_IMAGE_LAYOUT_GENERAL;
        }
        pe.output_layout_is_general = true;

        if (pe.output_resource == final_output_resource_) {
            final_pass_was_dirty = true;
        }
    }

    // 5. Blit final pass output → output_ (only if the output pass ran).
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


// ---------------------------------------------------------------------------
// Parameter SSBO upload
// ---------------------------------------------------------------------------
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
                                  uint32_t w, uint32_t h, uint64_t generation) {
    last_presented_pixels_ = pixels;
    last_presented_w_ = w;
    last_presented_h_ = h;
    last_presented_generation_ = generation;
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
    mark_downstream_dirty_(node_id);
}


void Engine::update_node_params_by_name(NodeId node_id,
                                        const std::unordered_map<std::string, float>& kv) {
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

    // Phase 6: warn on keys the manifest doesn't know about — these are
    // contract violations between Python and JSON.
    std::unordered_set<std::string> known;
    known.reserve(type->params.size());
    for (auto& p : type->params) known.insert(p.name);
    for (auto& [k, _v] : kv) {
        if (!known.count(k)) {
            log_warn("update_node_params_by_name: node " + std::to_string(node_id)
                   + " (type '" + type->id + "') has no param named '" + k
                   + "' — ignoring.");
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
                   + " — DROPPED (bug: GraphCompiler should have rejected this graph).");
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
    mark_downstream_dirty_(node_id);
}


bool Engine::upload_image(uint64_t node_id, const float* pixels,
                          uint32_t width, uint32_t height) {
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
        // Ring full — Python should retry next tick. Surface a soft failure.
        log_warn("upload_image: uploader ring full for node " + std::to_string(node_id));
        return false;
    }
    pending_uploads_.push_back({node_id, ticket});
    return true;
}


bool Engine::release_image(uint64_t node_id) {
    auto it = image_registry_.find(node_id);
    if (it == image_registry_.end()) return false;

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


} // namespace te