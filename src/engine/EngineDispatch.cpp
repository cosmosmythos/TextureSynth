#include "engine/Engine.hpp"
#include "engine/EngineBarriers.hpp"
#include "engine/Logging.hpp"
#include <algorithm>
#include <cstring>

namespace te {

void Engine::record_chain_dispatch_(VkCommandBuffer cmd, const PushConstants& pc,
                                    uint32_t gx, uint32_t gy, size_t chain_idx) {
    if (chain_idx >= chain_execs_.size()) return;
    const ChainExec& ce = chain_execs_[chain_idx];
    if (ce.member_pass_indices.empty()) return;
    const PassExec& head = passes_[ce.head_pass_index];
    const PassExec& tail = passes_[ce.tail_pass_index];

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
        // Also clear intermediate images for multi-pass chains.
        for (auto& intermedi : ce.intermediates) {
            if (intermedi.image && intermedi.image->image() != VK_NULL_HANDLE)
                vkCmdClearColorImage(cmd, intermedi.image->image(),
                                     VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        }
        return;
    }

    // Transition input images to general layout.
    for (const auto& inp : head.input_resources) {
        if (inp.node_id == 0) continue;
        auto* src = resources_.get(inp);
        if (!src) continue;
        if (src->layout == VK_IMAGE_LAYOUT_GENERAL &&
            !dirty_set_.is_dirty(inp.node_id)) continue;
        transition_output_to_general(cmd, src->image, src->layout, 0);
        src->layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    // Transition output image to general layout.
    for (uint32_t o = 0; o < tail.output_count; ++o) {
        auto* r = resources_.get(tail.output_resources[o]);
        if (!r) continue;
        transition_output_to_general(cmd, r->image, r->layout, r->alias_group_id);
        r->layout = VK_IMAGE_LAYOUT_GENERAL;
        if (r->alias_group_id > 0)
            resources_.record_alias_write(r->alias_group_id, *r);
    }

    VkDescriptorSet set = bindless_.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);

    // Multi-pass chain: loop over sub-passes with barriers between them.
    if (ce.sub_pass_count > 1 && !ce.sub_pipelines.empty()) {
        // Transition intermediates to GENERAL before first write (Bug 3 fix).
        for (uint32_t i = 0; i < ce.intermediates.size(); ++i) {
            if (ce.intermediates[i].image && ce.intermediates[i].image->image() != VK_NULL_HANDLE)
                transition_output_to_general(cmd, ce.intermediates[i].image->image(),
                                              VK_IMAGE_LAYOUT_UNDEFINED, 0);
        }
        for (uint32_t sp = 0; sp < ce.sub_pass_count && sp < (uint32_t)ce.sub_pipelines.size(); ++sp) {
            // Barrier: transition previous intermediate from storage write to sampled read.
            if (sp > 0) {
                uint32_t prev_idx = sp - 1;
                if (prev_idx < ce.intermediates.size() && ce.intermediates[prev_idx].image)
                    barrier_compute_write_to_sampled(cmd, ce.intermediates[prev_idx].image->image());
            }

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              ce.sub_pipelines[sp]->pipeline());

            PassPushConstants ppc{};
            ppc.global          = pc;
            ppc.param_base_slot = (uint32_t)head.param_base_slot;
            ppc.input_count     = head.input_count;
            ppc.pass_index      = sp;

            // External inputs: same for all sub-passes.
            for (uint32_t k = 0; k < ce.slot_source_count; ++k) {
                const auto& src = ce.slot_sources[k];
                uint32_t pass_idx = ce.member_pass_indices[src.member_idx];
                ppc.in_sampled_slots[k] = passes_[pass_idx].in_sampled_slots[src.input_index];
            }
            // Sub-passes > 0 must read from the previous intermediate, not the
            // original external input.  The intermediate was written by sub-pass
            // sp-1 and transitioned to SAMPLED_READ by the barrier above.
            if (sp > 0 && sp - 1 < ce.intermediates.size()
                && ce.intermediates[sp - 1].sampled_slot != BindlessTable::INVALID_SLOT) {
                ppc.in_sampled_slots[0] = ce.intermediates[sp - 1].sampled_slot;
            }
            for (uint32_t k = ce.slot_source_count; k < MAX_PASS_INPUTS; ++k)
                ppc.in_sampled_slots[k] = dummy_slot_;

            // Output: intermediate for non-last passes, final output for last.
            for (uint32_t t = 0; t < MAX_PASS_OUTPUTS; ++t)
                ppc.out_storage_slots[t] = ce.sub_out_storage_slots[sp][t];

            ppc.param_ring_idx = param_write_idx_;

            vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PassPushConstants), &ppc);
            auto& ts = ts_pools_[ts_pool_write_idx_];
            const uint32_t qi = ts_query_idx_; ts_query_idx_ += 2;
            if (ts.pool != VK_NULL_HANDLE && qi + 1 < ts.capacity) {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ts.pool, qi);
                vkCmdDispatch(cmd, gx, gy, 1);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ts.pool, qi + 1);
            } else {
                vkCmdDispatch(cmd, gx, gy, 1);
            }
            ++last_dispatch_count_;
        }
        return;
    }

    // Single-pipeline chain: original path.
    if (!ce.pipeline) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ce.pipeline->pipeline());

    PassPushConstants ppc{};
    ppc.global           = pc;
    ppc.param_base_slot  = (uint32_t)head.param_base_slot;
    ppc.input_count      = head.input_count;
    ppc.pass_index       = 0;
    for (uint32_t k = 0; k < ce.slot_source_count; ++k) {
        const auto& src = ce.slot_sources[k];
        uint32_t pass_idx = ce.member_pass_indices[src.member_idx];
        ppc.in_sampled_slots[k] = passes_[pass_idx].in_sampled_slots[src.input_index];
    }
    for (uint32_t k = ce.slot_source_count; k < MAX_PASS_INPUTS; ++k)
        ppc.in_sampled_slots[k] = dummy_slot_;
    for (uint32_t t = 0; t < MAX_PASS_OUTPUTS; ++t)
        ppc.out_storage_slots[t] = tail.out_storage_slots[t];
    ppc.param_ring_idx = param_write_idx_;

    vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PassPushConstants), &ppc);
    auto& ts = ts_pools_[ts_pool_write_idx_];
    const uint32_t qi = ts_query_idx_; ts_query_idx_ += 2;
    if (ts.pool != VK_NULL_HANDLE && qi + 1 < ts.capacity) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ts.pool, qi);
        vkCmdDispatch(cmd, gx, gy, 1);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ts.pool, qi + 1);
    } else {
        vkCmdDispatch(cmd, gx, gy, 1);
    }
    ++last_dispatch_count_;
}


void Engine::record_dispatch(VkCommandBuffer cmd, const PushConstants& pc) {
    last_dispatch_count_ = 0;
    if (passes_.empty()) return;

    if (param_dirty_) {
        const uint32_t next = (param_write_idx_ + 1) % PARAM_RING;
        std::memcpy(param_mapped_[next], param_mapped_[param_write_idx_], MAX_NODE_PARAMS * sizeof(float));
        vmaFlushAllocation(ctx_.allocator(), param_alloc_[next], 0, VK_WHOLE_SIZE);

        param_write_idx_ = next;
        param_dirty_ = false;

        // Make host writes to param SSBO visible to compute shader reads.
        // vmaFlushAllocation only flushes the CPU cache; a pipeline barrier
        // is required to ensure the GPU sees the updated data.
        {
            VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
            barrier.srcStageMask  = VK_PIPELINE_STAGE_2_HOST_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT;
            barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers    = &barrier;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
    }

    const uint32_t gx = (pc.resolution_x + 7) / 8;
    const uint32_t gy = (pc.resolution_y + 7) / 8;
    dirty_set_.propagate(downstream_adj_);
    if (!dirty_set_.any()) return;

    ts_pool_write_idx_ = param_write_idx_ % TIMESTAMP_POOL_COUNT;
    ts_query_idx_ = 1;
    auto& ts_pool = ts_pools_[ts_pool_write_idx_];
    if (ts_pool.pool != VK_NULL_HANDLE && ts_pool.capacity > 0) {
        vkCmdResetQueryPool(cmd, ts_pool.pool, 0, ts_pool.capacity);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ts_pool.pool, 0);
    }

    resolve_aliased_staleness_();
    record_barriers_(cmd, pc);
    bool final_pass_was_dirty = record_pass_dispatches_(cmd, pc, gx, gy);
    record_final_copy_(cmd, pc, final_pass_was_dirty);
}


void Engine::record_barriers_(VkCommandBuffer cmd, const PushConstants&) {
    for (auto& pe : passes_) {
        if (!dirty_set_.is_dirty(pe.node_id)) continue;
        for (uint32_t i = 0; i < pe.output_count; ++i) {
            const ResourceUUID& rid = pe.output_resources[i];
            auto* r = resources_.get(rid);
            if (!r) continue;
            transition_output_to_general(cmd, r->image, r->layout, r->alias_group_id);
            r->layout = VK_IMAGE_LAYOUT_GENERAL;
            if (r->alias_group_id > 0)
                resources_.record_alias_write(r->alias_group_id, *r);
        }
    }

    if (dummy_layout_ != VK_IMAGE_LAYOUT_GENERAL) {
        transition(cmd, dummy_image_.image(),
                   dummy_layout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        dummy_layout_ = VK_IMAGE_LAYOUT_GENERAL;
    }

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
            acq.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
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

    if (output_layout_ != VK_IMAGE_LAYOUT_GENERAL) {
        transition(cmd, output_storage_->image(),
                   output_layout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        output_layout_ = VK_IMAGE_LAYOUT_GENERAL;
    }
}


bool Engine::record_pass_dispatches_(VkCommandBuffer cmd, const PushConstants& pc,
                                     uint32_t gx, uint32_t gy) {
    bool final_pass_was_dirty = false;

    // Phase 1: Dispatch all chains in chain_execs_ index order (topologically sorted
    // by FusedGraphCompiler). This ensures a producer chain always dispatches before
    // its consumer chains, regardless of pass-index order.
    for (size_t cid = 0; cid < chain_execs_.size(); ++cid) {
        bool chain_dirty = dirty_set_.is_chain_dirty(static_cast<ChainId>(cid));
        if (!chain_dirty) continue;
        // Barrier: ensure prior dispatches' storage writes to cross-chain input
        // images are visible as sampled reads in this chain.  Without this,
        // the consumer chain may read stale/undefined data from producer chains.
        const auto& ce_pre = chain_execs_[cid];
        for (uint32_t k = 0; k < ce_pre.slot_source_count; ++k) {
            const auto& src = ce_pre.slot_sources[k];
            uint32_t pass_idx = ce_pre.member_pass_indices[src.member_idx];
            if (pass_idx >= passes_.size()) continue;
            const auto& inp = passes_[pass_idx].input_resources[src.input_index];
            if (inp.node_id == 0) continue;
            auto* r = resources_.get(inp);
            if (!r) continue;
            if (r->layout == VK_IMAGE_LAYOUT_GENERAL &&
                !dirty_set_.is_dirty(inp.node_id))
                continue;
            barrier_compute_write_to_sampled(cmd, r->image);
        }
        const size_t tail_i = chain_execs_[cid].tail_pass_index;
        bool tail_has_final = false;
        if (tail_i < passes_.size()) {
            const auto& tail_pe = passes_[tail_i];
            tail_has_final = (std::find(tail_pe.output_resources.begin(),
                              tail_pe.output_resources.end(),
                              final_output_resource_) != tail_pe.output_resources.end());
        }
        if (tail_has_final) final_pass_was_dirty = true;
        record_chain_dispatch_(cmd, pc, gx, gy, cid);
    }

    // Phase 2: Dispatch non-chain (solo) passes in pass-index order.
    for (size_t i = 0; i < passes_.size(); ++i) {
        auto& pe = passes_[i];
        bool is_chain_member = (i < chain_id_of_pass_.size() &&
                                chain_id_of_pass_[i] != UINT32_MAX);
        if (is_chain_member) continue;
        if (!dirty_set_.is_dirty(pe.node_id)) continue;

        if (!pe.bypassed) {
            for (const auto& inp : pe.input_resources) {
                if (inp.node_id == 0) continue;
                auto* src = resources_.get(inp);
                if (!src) continue;
                if (src->layout == VK_IMAGE_LAYOUT_GENERAL &&
                    !dirty_set_.is_dirty(inp.node_id)) {
                    continue;
                }
                barrier_compute_write_to_sampled(cmd, src->image);
            }
        }

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
            ppc.pass_index       = 0;
            for (uint32_t k = 0; k < MAX_PASS_INPUTS; ++k)
                ppc.in_sampled_slots[k] = pe.in_sampled_slots[k];
            for (uint32_t t = 0; t < MAX_PASS_OUTPUTS; ++t)
                ppc.out_storage_slots[t] = pe.out_storage_slots[t];
            ppc.param_ring_idx = param_write_idx_;

            vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PassPushConstants), &ppc);
            auto& ts = ts_pools_[ts_pool_write_idx_];
            const uint32_t qi = ts_query_idx_; ts_query_idx_ += 2;
            if (ts.pool != VK_NULL_HANDLE && qi + 1 < ts.capacity) {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ts.pool, qi);
                vkCmdDispatch(cmd, gx, gy, 1);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ts.pool, qi + 1);
            } else {
                vkCmdDispatch(cmd, gx, gy, 1);
            }
            ++last_dispatch_count_;
        }

        if (std::find(pe.output_resources.begin(), pe.output_resources.end(), final_output_resource_) != pe.output_resources.end()) {
            final_pass_was_dirty = true;
        }
    }

    {
        auto& ts = ts_pools_[ts_pool_write_idx_];
        const uint32_t qi = ts_query_idx_;
        if (ts.pool != VK_NULL_HANDLE && qi < ts.capacity) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, ts.pool, qi);
        }
    }

    return final_pass_was_dirty;
}


void Engine::record_final_copy_(VkCommandBuffer cmd, const PushConstants& pc,
                                bool final_pass_was_dirty) {
    auto* final_res = resources_.get(final_output_resource_);
    if (!final_res || !final_pass_was_dirty || !final_copy_pipeline_) return;

    uint32_t sampled_slot = BindlessTable::INVALID_SLOT;
    auto sit = res_sampled_slot_.find(final_output_resource_);
    if (sit == res_sampled_slot_.end()) {
        sampled_slot = bindless_.alloc_sampled_slot();
        if (sampled_slot == BindlessTable::INVALID_SLOT) {
            log_error("record_final_copy_: sampled bindless slot allocation failed");
            return;
        }
        res_sampled_slot_[final_output_resource_] = sampled_slot;
        bindless_.write_sampled(ctx_, sampled_slot, final_res->view, VK_IMAGE_LAYOUT_GENERAL);
    } else {
        sampled_slot = sit->second;
    }

    barrier_compute_write_to_sampled_read(cmd, final_res->image);
    final_res->layout = VK_IMAGE_LAYOUT_GENERAL;

    if (output_layout_ != VK_IMAGE_LAYOUT_GENERAL) {
        transition(cmd, output_storage_->image(),
                   output_layout_, VK_IMAGE_LAYOUT_GENERAL,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        output_layout_ = VK_IMAGE_LAYOUT_GENERAL;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                      final_copy_pipeline_->pipeline());
    VkDescriptorSet set = bindless_.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);

    PassPushConstants ppc{};
    ppc.global = pc;
    ppc.global.resolution_x = output_w_;
    ppc.global.resolution_y = output_h_;
    ppc.out_storage_slots[0] = output_storage_slot_;
    ppc.input_count = 1;
    ppc.in_sampled_slots[0] = sampled_slot;
    for (uint32_t k = 1; k < MAX_PASS_INPUTS; ++k)
        ppc.in_sampled_slots[k] = dummy_slot_;
    ppc.param_ring_idx = param_write_idx_;

    vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PassPushConstants), &ppc);
    vkCmdDispatch(cmd, (output_w_ + 7) / 8, (output_h_ + 7) / 8, 1);
    output_layout_ = VK_IMAGE_LAYOUT_GENERAL;
}

} // namespace te
