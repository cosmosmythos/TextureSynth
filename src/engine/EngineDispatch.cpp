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
        return;
    }
    if (!ce.pipeline) return;

    for (const auto& inp : head.input_resources) {
        if (inp.node_id == 0) continue;
        auto* src = resources_.get(inp);
        if (!src) continue;
        if (src->layout == VK_IMAGE_LAYOUT_GENERAL &&
            !dirty_set_.is_dirty(inp.node_id)) continue;
        barrier_compute_write_to_sampled_read(cmd, src->image);
        src->layout = VK_IMAGE_LAYOUT_GENERAL;
    }

    for (uint32_t o = 0; o < tail.output_count; ++o) {
        auto* r = resources_.get(tail.output_resources[o]);
        if (!r) continue;
        transition_output_to_general(cmd, r->image, r->layout, r->alias_group_id);
        r->layout = VK_IMAGE_LAYOUT_GENERAL;
        if (r->alias_group_id > 0)
            resources_.record_alias_write(r->alias_group_id, *r);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ce.pipeline->pipeline());
    VkDescriptorSet set = bindless_.set();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            bindless_.pipeline_layout(), 0, 1, &set, 0, nullptr);

    PassPushConstants ppc{};
    ppc.global           = pc;
    ppc.param_base_slot  = (uint32_t)head.param_base_slot;
    ppc.input_count      = head.input_count;
    for (uint32_t k = 0; k < MAX_PASS_INPUTS; ++k)
        ppc.in_sampled_slots[k] = ce.chain_in_sampled_slots[k];
    for (uint32_t t = 0; t < MAX_PASS_OUTPUTS; ++t)
        ppc.out_storage_slots[t] = tail.out_storage_slots[t];
    ppc.param_ring_idx = param_write_idx_;

    // DIAG: print push constants for chains with external inputs
    {
        bool has_ext = false;
        for (uint32_t k = 0; k < MAX_PASS_INPUTS; ++k)
            if (ce.chain_in_sampled_slots[k] != 0) { has_ext = true; break; }
        if (has_ext) {
            std::string msg = "[DIAG] DISPATCH chain=" + std::to_string(chain_idx)
                + " head_node=" + std::to_string(head.node_id)
                + " tail_node=" + std::to_string(tail.node_id)
                + " in_count=" + std::to_string(head.input_count)
                + " in_sampled=[";
            for (uint32_t k = 0; k < head.input_count && k < MAX_PASS_INPUTS; ++k) {
                msg += std::to_string(ppc.in_sampled_slots[k]);
                if (k + 1 < head.input_count && k + 1 < MAX_PASS_INPUTS) msg += ", ";
            }
            msg += "] out_storage=[";
            for (uint32_t t = 0; t < tail.output_count && t < MAX_PASS_OUTPUTS; ++t) {
                msg += std::to_string(ppc.out_storage_slots[t]);
                if (t + 1 < tail.output_count && t + 1 < MAX_PASS_OUTPUTS) msg += ", ";
            }
            msg += "] param_base=" + std::to_string(ppc.param_base_slot);
            log_info(msg);
        }
    }

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
    record_final_blit_(cmd, pc, final_pass_was_dirty);
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
        for (int i = 0; i < 6; ++i) {
            transition(cmd, dummy_images_[i].image(),
                       dummy_layout_, VK_IMAGE_LAYOUT_GENERAL,
                       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                       VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }
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
            acq.oldLayout     = VK_IMAGE_LAYOUT_GENERAL;
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
    std::vector<uint8_t> chain_dispatched(chain_execs_.size(), 0);

    // DIAG: print dispatch order
    {
        std::string msg = "[DIAG] DISPATCH ORDER: " + std::to_string(passes_.size())
            + " passes, " + std::to_string(chain_execs_.size()) + " chains";
        log_info(msg);
        for (size_t i = 0; i < passes_.size(); ++i) {
            const auto& pe = passes_[i];
            bool is_chain = (i < chain_id_of_pass_.size() && chain_id_of_pass_[i] != UINT32_MAX);
            uint32_t cid = is_chain ? chain_id_of_pass_[i] : UINT32_MAX;
            log_info("[DIAG]   pass[" + std::to_string(i) + "] node=" + std::to_string(pe.node_id)
                     + (is_chain ? (" chain=" + std::to_string(cid)) : " [solo]"));
        }
    }

    for (size_t i = 0; i < passes_.size(); ++i) {
        auto& pe = passes_[i];
        bool is_chain_member = (i < chain_id_of_pass_.size() &&
                                chain_id_of_pass_[i] != UINT32_MAX);
        if (is_chain_member) {
            ChainId cid = (ChainId)chain_id_of_pass_[i];
            if (!chain_dispatched[cid]) {
                chain_dispatched[cid] = 1;
                if (dirty_set_.is_chain_dirty(cid)) {
                    const size_t tail_i = chain_execs_[cid].tail_pass_index;
                    if (tail_i < passes_.size()) {
                        const auto& tail_pe = passes_[tail_i];
                        if (std::find(tail_pe.output_resources.begin(),
                                      tail_pe.output_resources.end(),
                                      final_output_resource_) != tail_pe.output_resources.end()) {
                            final_pass_was_dirty = true;
                        }
                    }
                    record_chain_dispatch_(cmd, pc, gx, gy, cid);
                }
            }
            continue;
        }
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


void Engine::record_final_blit_(VkCommandBuffer cmd, const PushConstants& pc,
                                bool final_pass_was_dirty) {
    auto* final_res = resources_.get(final_output_resource_);
    if (!final_res || !final_pass_was_dirty) return;

    transition(cmd, final_res->image,
               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
               VK_ACCESS_2_TRANSFER_READ_BIT);
    final_res->layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    transition(cmd, output_storage_->image(),
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

    transition(cmd, output_storage_->image(),
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
               VK_ACCESS_2_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    output_layout_ = VK_IMAGE_LAYOUT_GENERAL;

    transition(cmd, final_res->image,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
               VK_PIPELINE_STAGE_2_TRANSFER_BIT,
               VK_ACCESS_2_TRANSFER_READ_BIT,
               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
               VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    final_res->layout = VK_IMAGE_LAYOUT_GENERAL;
}

} // namespace te
