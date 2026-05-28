#include "engine/AsyncReadback.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include "engine/Logging.hpp"
#include <cstring>
#include <mutex>

namespace te {

bool AsyncReadback::init(VulkanContext& ctx, uint32_t max_w, uint32_t max_h) {
    current_capacity_w_ = max_w;
    current_capacity_h_ = max_h;
    const VkDeviceSize cap = (VkDeviceSize)max_w * max_h * 4 * sizeof(float);
    for (auto& s : slots_) {
        if (!init_slot_(ctx, s, cap)) {
            log_error("AsyncReadback: slot init failed");
            return false;
        }
    }
    log_info("AsyncReadback: initialized " + std::to_string(SLOT_COUNT)
             + " slots @ " + std::to_string(max_w) + "x" + std::to_string(max_h));
    return true;
}

bool AsyncReadback::init_slot_(VulkanContext& ctx, Slot& s, VkDeviceSize capacity) {
    // Command pool (resettable so we can re-record each submit).
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = ctx.graphics_family();
    if (vkCreateCommandPool(ctx.device(), &pci, nullptr, &s.pool) != VK_SUCCESS)
        return false;

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = s.pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx.device(), &cai, &s.cmd) != VK_SUCCESS)
        return false;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    // NOT signaled: a freshly-created fence with state=Free should not be
    // mistakenly considered "complete".
    if (vkCreateFence(ctx.device(), &fci, nullptr, &s.fence) != VK_SUCCESS)
        return false;

    return grow_staging_(ctx, s, capacity);
}

bool AsyncReadback::grow_staging_(VulkanContext& ctx, Slot& s, VkDeviceSize new_capacity) {
    if (s.staging) {
        vmaDestroyBuffer(ctx.allocator(), s.staging, s.staging_alloc);
        s.staging = VK_NULL_HANDLE;
        s.staging_alloc = nullptr;
        s.staging_mapped = nullptr;
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = new_capacity;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo ainfo{};
    if (vmaCreateBuffer(ctx.allocator(), &bci, &aci,
                        &s.staging, &s.staging_alloc, &ainfo) != VK_SUCCESS) {
        log_error("AsyncReadback: staging alloc failed");
        return false;
    }
    s.staging_mapped   = ainfo.pMappedData;
    s.staging_capacity = new_capacity;
    return true;
}

void AsyncReadback::destroy_slot_(VulkanContext& ctx, Slot& s) {
    if (s.staging) {
        vmaDestroyBuffer(ctx.allocator(), s.staging, s.staging_alloc);
        s.staging = VK_NULL_HANDLE;
        s.staging_alloc = nullptr;
        s.staging_mapped = nullptr;
    }
    if (s.fence) { vkDestroyFence(ctx.device(), s.fence, nullptr); s.fence = VK_NULL_HANDLE; }
    if (s.pool)  { vkDestroyCommandPool(ctx.device(), s.pool, nullptr); s.pool = VK_NULL_HANDLE; s.cmd = VK_NULL_HANDLE; }
}

void AsyncReadback::shutdown(VulkanContext& ctx) {
    // Drain any in-flight work first.
    for (auto& s : slots_) {
        if (s.state == SlotState::InFlight && s.fence) {
            vkWaitForFences(ctx.device(), 1, &s.fence, VK_TRUE, UINT64_MAX);
        }
    }
    for (auto& s : slots_) destroy_slot_(ctx, s);
}

bool AsyncReadback::ensure_capacity(VulkanContext& ctx, uint32_t w, uint32_t h) {
    if (w <= current_capacity_w_ && h <= current_capacity_h_) return true;
    // Drain so we can safely resize staging buffers.
    for (auto& s : slots_) {
        if (s.state == SlotState::InFlight) {
            vkWaitForFences(ctx.device(), 1, &s.fence, VK_TRUE, UINT64_MAX);
            s.state = SlotState::Free;
        }
    }
    const VkDeviceSize new_cap = (VkDeviceSize)w * h * 4 * sizeof(float);
    for (auto& s : slots_) {
        if (!grow_staging_(ctx, s, new_cap)) return false;
    }
    current_capacity_w_ = w;
    current_capacity_h_ = h;
    log_info("AsyncReadback: resized staging → " + std::to_string(w) + "x" + std::to_string(h));
    return true;
}

bool AsyncReadback::any_in_flight() const {
    for (auto& s : slots_) if (s.state == SlotState::InFlight) return true;
    return false;
}

void AsyncReadback::drain(VulkanContext& ctx) {
    for (auto& s : slots_) {
        if (s.state == SlotState::InFlight && s.fence) {
            vkWaitForFences(ctx.device(), 1, &s.fence, VK_TRUE, UINT64_MAX);
            s.state  = SlotState::Free;
            s.ticket = 0;
        }
    }
    synthetic_ready_ = false;
    synthetic_pixels_.clear();
}

uint64_t AsyncReadback::publish_synthetic(const std::vector<float>& pixels, uint32_t w, uint32_t h, uint64_t generation) {
    synthetic_pixels_ = pixels;
    synthetic_w_ = w;
    synthetic_h_ = h;
    synthetic_generation_ = generation;
    synthetic_ticket_ = next_ticket_++;
    synthetic_ready_ = true;
    return synthetic_ticket_;
}

uint64_t AsyncReadback::submit(VulkanContext& ctx, Engine& engine, const PushConstants& pc, uint64_t generation) {
    // ─── Short-circuit when nothing changed ───────────────────────────
    if (!engine.any_pass_dirty() && engine.has_presented_frame()) {
        return engine.republish_last_frame(generation);
    }

    Slot* slot = nullptr;
    for (auto& s : slots_) {
        if (s.state == SlotState::Free) { slot = &s; break; }
    }
    if (!slot) return 0;

    const uint32_t w = engine.output().extent().width;
    const uint32_t h = engine.output().extent().height;
    if (!ensure_capacity(ctx, w, h)) return 0;

    if (vkResetCommandBuffer(slot->cmd, 0) != VK_SUCCESS) {
        log_error("AsyncReadback: vkResetCommandBuffer failed");
        return 0;
    }
    if (vkResetFences(ctx.device(), 1, &slot->fence) != VK_SUCCESS) {
        log_error("AsyncReadback: vkResetFences failed");
        return 0;
    }

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->cmd, &bi) != VK_SUCCESS) {
        log_error("AsyncReadback: vkBeginCommandBuffer failed");
        return 0;
    }

    auto barrier = [&](VkImageLayout from, VkImageLayout to,
                       VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                       VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = src_stage; b.srcAccessMask = src_access;
        b.dstStageMask = dst_stage; b.dstAccessMask = dst_access;
        b.oldLayout = from; b.newLayout = to;
        b.image = engine.output().image();
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(slot->cmd, &dep);
    };

    engine.record_dispatch(slot->cmd, pc);

    barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent      = {w, h, 1};
    vkCmdCopyImageToBuffer(slot->cmd, engine.output().image(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           slot->staging, 1, &copy);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, 0);

    if (vkEndCommandBuffer(slot->cmd) != VK_SUCCESS) {
        log_error("AsyncReadback: vkEndCommandBuffer failed");
        return 0;
    }

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &slot->cmd;
    {
        std::lock_guard<std::mutex> lk(ctx.graphics_queue_mutex());
        if (vkQueueSubmit(ctx.graphics_queue(), 1, &si, slot->fence) != VK_SUCCESS) {
            log_error("AsyncReadback: vkQueueSubmit failed");
            return 0;
        }
    }

    slot->state          = SlotState::InFlight;
    slot->job_w          = w;
    slot->job_h          = h;
    slot->job_generation = generation;
    slot->ticket         = next_ticket_++;
    if (generation > latest_submitted_generation_)
        latest_submitted_generation_ = generation;

    engine.mark_all_clean();
    return slot->ticket;
}

bool AsyncReadback::poll(VulkanContext& ctx,
                         std::vector<float>& out_pixels,
                         uint32_t& out_w, uint32_t& out_h,
                         uint64_t& out_generation) {
    // ── Synthetic (CPU-cached) frame has highest priority ────────────
    if (synthetic_ready_) {
        out_pixels     = std::move(synthetic_pixels_);
        out_w          = synthetic_w_;
        out_h          = synthetic_h_;
        out_generation = synthetic_generation_;
        synthetic_ready_ = false;
        synthetic_pixels_.clear();
        return true;
    }

    // Recycle finished slots whose generation is older than the latest submission.
    for (auto& s : slots_) {
        if (s.state != SlotState::InFlight) continue;
        if (vkGetFenceStatus(ctx.device(), s.fence) != VK_SUCCESS) continue;
        if (s.job_generation != 0 &&
            s.job_generation < latest_submitted_generation_) {
            s.state  = SlotState::Free;
            s.ticket = 0;
        }
    }

    // Find the *newest* finished slot among the remaining ones.
    Slot* newest = nullptr;
    for (auto& s : slots_) {
        if (s.state != SlotState::InFlight) continue;
        if (vkGetFenceStatus(ctx.device(), s.fence) != VK_SUCCESS) continue;
        if (!newest || s.ticket > newest->ticket) newest = &s;
    }
    if (!newest) return false;

    // Free older finished slots (we're skipping their pixels — they're stale).
    for (auto& s : slots_) {
        if (&s == newest) continue;
        if (s.state == SlotState::InFlight &&
            vkGetFenceStatus(ctx.device(), s.fence) == VK_SUCCESS) {
            s.state  = SlotState::Free;
            s.ticket = 0;
        }
    }

    vmaInvalidateAllocation(ctx.allocator(), newest->staging_alloc, 0, VK_WHOLE_SIZE);

    const size_t n_floats = (size_t)newest->job_w * newest->job_h * 4;
    out_pixels.resize(n_floats);
    std::memcpy(out_pixels.data(), newest->staging_mapped, n_floats * sizeof(float));

    out_w          = newest->job_w;
    out_h          = newest->job_h;
    out_generation = newest->job_generation;

    newest->state  = SlotState::Free;
    newest->ticket = 0;
    return true;
}

} // namespace te
