#include "engine/Engine.hpp"
#include "engine/EngineBarriers.hpp"
#include "engine/Logging.hpp"
#include <algorithm>
#include <cstring>
#include <cstdlib>

namespace te {

void Engine::record_dispatch(VkCommandBuffer cmd, const PushConstants& pc) {
    last_dispatch_count_ = 0;
    if (group_execs_.empty()) return;

    if (param_dirty_) {
        const uint32_t next = (param_write_idx_ + 1) % PARAM_RING;
        std::memcpy(param_mapped_[next], param_mapped_[param_write_idx_], MAX_NODE_PARAMS * sizeof(float));
        vmaFlushAllocation(ctx_.allocator(), param_alloc_[next], 0, VK_WHOLE_SIZE);

        param_write_idx_ = next;
        param_dirty_ = false;

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

    dirty_set_.propagate(downstream_adj_);
    if (!dirty_set_.any()) return;

    ts_pool_write_idx_ = param_write_idx_ % TIMESTAMP_POOL_COUNT;
    ts_query_idx_ = 1;
    auto& ts_pool = ts_pools_[ts_pool_write_idx_];
    if (ts_pool.pool != VK_NULL_HANDLE && ts_pool.capacity > 0) {
        vkCmdResetQueryPool(cmd, ts_pool.pool, 0, ts_pool.capacity);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, ts_pool.pool, 1);
    }

    resolve_aliased_staleness_();
    record_barriers_(cmd, pc);

    record_group_dispatches_(cmd, pc);

    // Copy last group's output to output_storage_ via final_copy pipeline.
    auto& last = group_execs_.back();
    if (last.output_image && last.output_image->image() != VK_NULL_HANDLE
        && final_copy_pipeline_) {
        barrier_compute_write_to_sampled_read(cmd, last.output_image->image());

        uint32_t sampled = bindless_.alloc_sampled_slot();
        if (sampled != BindlessTable::INVALID_SLOT) {
            bindless_.write_sampled(ctx_, sampled,
                                    last.output_image->view(), VK_IMAGE_LAYOUT_GENERAL);

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

            PassPushConstants pass_push_constants{};
            pass_push_constants.global = pc;
            pass_push_constants.global.resolution_x = output_w_;
            pass_push_constants.global.resolution_y = output_h_;
            pass_push_constants.out_storage_slots[0] = output_storage_slot_;
            pass_push_constants.input_count = 1;
            pass_push_constants.in_sampled_slots[0] = sampled;
            for (uint32_t k = 1; k < MAX_PASS_INPUTS; ++k)
                pass_push_constants.in_sampled_slots[k] = dummy_slot_;
            pass_push_constants.param_ring_idx = param_write_idx_;

            vkCmdPushConstants(cmd, bindless_.pipeline_layout(),
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(PassPushConstants), &pass_push_constants);
            vkCmdDispatch(cmd, (output_w_ + 7) / 8, (output_h_ + 7) / 8, 1);

            bindless_.free_sampled_slot(sampled);
        }
    }

    // End timestamp for GPU timing.
    {
        auto& ts_pool = ts_pools_[ts_pool_write_idx_];
        if (ts_pool.pool != VK_NULL_HANDLE && ts_pool.capacity > 2) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                 ts_pool.pool, 2);
        }
    }
}


void Engine::record_barriers_(VkCommandBuffer cmd, const PushConstants&) {
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

} // namespace te
