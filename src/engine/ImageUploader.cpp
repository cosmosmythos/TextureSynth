#include "engine/ImageUploader.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
#include <cstring>
#include <mutex>

namespace te {

bool ImageUploader::init(VulkanContext& ctx) {
    for (auto& s : slots_) {
        if (!init_slot_(ctx, s)) {
            log_error("ImageUploader: slot init failed");
            return false;
        }
    }
    log_info("ImageUploader: " + std::to_string(SLOT_COUNT)
             + " slots on transfer_family=" + std::to_string(ctx.transfer_family()));
    return true;
}

bool ImageUploader::init_slot_(VulkanContext& ctx, Slot& s) {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
                         | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = ctx.transfer_family();
    if (vkCreateCommandPool(ctx.device(), &pci, nullptr, &s.pool) != VK_SUCCESS)
        return false;

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = s.pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx.device(), &cai, &s.cmd) != VK_SUCCESS)
        return false;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(ctx.device(), &fci, nullptr, &s.fence) != VK_SUCCESS)
        return false;

    // Seed with 1024x1024 RGBA32F = 16 MB. Grows on demand.
    return grow_staging_(ctx, s, (VkDeviceSize)1024 * 1024 * 16);
}

bool ImageUploader::grow_staging_(VulkanContext& ctx, Slot& s, VkDeviceSize cap) {
    if (s.staging) {
        vmaDestroyBuffer(ctx.allocator(), s.staging, s.staging_alloc);
        s.staging = VK_NULL_HANDLE;
        s.staging_alloc = nullptr;
        s.staging_mapped = nullptr;
    }
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = cap;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
              | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ai{};
    if (vmaCreateBuffer(ctx.allocator(), &bci, &aci,
                        &s.staging, &s.staging_alloc, &ai) != VK_SUCCESS) {
        log_error("ImageUploader: staging alloc failed");
        return false;
    }
    s.staging_mapped   = ai.pMappedData;
    s.staging_capacity = cap;
    return true;
}

void ImageUploader::destroy_slot_(VulkanContext& ctx, Slot& s) {
    if (s.image) { s.image->destroy(ctx); s.image.reset(); }
    if (s.staging) {
        vmaDestroyBuffer(ctx.allocator(), s.staging, s.staging_alloc);
        s.staging = VK_NULL_HANDLE;
        s.staging_alloc = nullptr;
        s.staging_mapped = nullptr;
    }
    if (s.fence) { vkDestroyFence(ctx.device(), s.fence, nullptr); s.fence = VK_NULL_HANDLE; }
    if (s.pool)  { vkDestroyCommandPool(ctx.device(), s.pool, nullptr); s.pool = VK_NULL_HANDLE; }
    s.cmd = VK_NULL_HANDLE;
}

void ImageUploader::shutdown(VulkanContext& ctx) {
    drain(ctx);
    for (auto& s : slots_) destroy_slot_(ctx, s);
}

void ImageUploader::drain(VulkanContext& ctx) {
    for (auto& s : slots_) {
        if (s.state == SlotState::InFlight && s.fence) {
            vkWaitForFences(ctx.device(), 1, &s.fence, VK_TRUE, UINT64_MAX);
            s.state = SlotState::Free;
            if (s.image) { s.image->destroy(ctx); s.image.reset(); }
            s.ticket = 0;
            s.needs_acquire_on_graphics = false;
        }
    }
}

bool ImageUploader::any_in_flight() const {
    for (auto& s : slots_) if (s.state == SlotState::InFlight) return true;
    return false;
}

uint64_t ImageUploader::submit(VulkanContext& ctx,
                               uint64_t node_id,
                               const float* pixels, uint32_t w, uint32_t h,
                               std::unique_ptr<Image> recycled_image) {
    Slot* slot = nullptr;
    for (auto& s : slots_) if (s.state == SlotState::Free) { slot = &s; break; }
    if (!slot) return 0;  // ring full — caller retries next tick

    const VkDeviceSize needed = (VkDeviceSize)w * h * 4 * sizeof(float);
    if (needed > slot->staging_capacity) {
        if (!grow_staging_(ctx, *slot, needed)) return 0;
    }

    // Create or reuse destination image.
    std::unique_ptr<Image> img = std::move(recycled_image);
    if (!img) img = std::make_unique<Image>();
    if (img->extent().width != w || img->extent().height != h ||
        img->image() == VK_NULL_HANDLE) {
        img->destroy(ctx);
        if (!img->create(ctx, w, h, VK_FORMAT_R32G32B32A32_SFLOAT,
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            log_error("ImageUploader: image create failed");
            return 0;
        }
    }

    // Copy CPU → mapped staging (no Vulkan calls).
    std::memcpy(slot->staging_mapped, pixels, (size_t)needed);
    vmaFlushAllocation(ctx.allocator(), slot->staging_alloc, 0, needed);

    if (vkResetCommandBuffer(slot->cmd, 0) != VK_SUCCESS) return 0;
    if (vkResetFences(ctx.device(), 1, &slot->fence) != VK_SUCCESS) return 0;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(slot->cmd, &bi) != VK_SUCCESS) return 0;

    const bool qfot = (ctx.transfer_family() != ctx.graphics_family());

    // 1. UNDEFINED → TRANSFER_DST_OPTIMAL (on transfer queue).
    VkImageMemoryBarrier2 to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    to_dst.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_dst.srcAccessMask = 0;
    to_dst.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
    to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_dst.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_dst.image         = img->image();
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &to_dst;
    vkCmdPipelineBarrier2(slot->cmd, &dep);

    // 2. Buffer → image copy.
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent      = {w, h, 1};
    vkCmdCopyBufferToImage(slot->cmd, slot->staging, img->image(),
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    // 3. Release / final transition.
    if (qfot) {
        // Queue family ownership release: transfer -> graphics. Release has no dstStage; acquire has no srcStage.
        VkImageMemoryBarrier2 rel{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        rel.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        rel.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        rel.dstStageMask  = VK_PIPELINE_STAGE_2_NONE;   // release: no dst on this queue
        rel.dstAccessMask = 0;
        rel.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        rel.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        rel.srcQueueFamilyIndex = ctx.transfer_family();
        rel.dstQueueFamilyIndex = ctx.graphics_family();
        rel.image         = img->image();
        rel.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        dep.pImageMemoryBarriers = &rel;
        vkCmdPipelineBarrier2(slot->cmd, &dep);
        slot->needs_acquire_on_graphics = true;
    } else {
        // Same queue — direct transition to GENERAL is enough.
        VkImageMemoryBarrier2 to_gen{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        to_gen.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
        to_gen.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_gen.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        to_gen.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                             | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        to_gen.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_gen.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
        to_gen.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_gen.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_gen.image         = img->image();
        to_gen.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        dep.pImageMemoryBarriers = &to_gen;
        vkCmdPipelineBarrier2(slot->cmd, &dep);
        slot->needs_acquire_on_graphics = false;
    }

    if (vkEndCommandBuffer(slot->cmd) != VK_SUCCESS) return 0;

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &slot->cmd;
    {
        std::lock_guard<std::mutex> lk(ctx.transfer_queue_mutex());
        if (vkQueueSubmit(ctx.transfer_queue(), 1, &si, slot->fence) != VK_SUCCESS) {
            log_error("ImageUploader: vkQueueSubmit failed");
            return 0;
        }
    }

    slot->state   = SlotState::InFlight;
    slot->image   = std::move(img);
    slot->node_id = node_id;
    slot->ticket  = next_ticket_++;
    slot->job_w   = w;
    slot->job_h   = h;
    return slot->ticket;
}

std::vector<ImageUploader::Completion> ImageUploader::poll(VulkanContext& ctx) {
    std::vector<Completion> out;
    for (auto& s : slots_) {
        if (s.state != SlotState::InFlight) continue;
        if (vkGetFenceStatus(ctx.device(), s.fence) != VK_SUCCESS) continue;

        // If QFOT release was issued on transfer queue, acquire on graphics queue is done lazily in Engine::record_dispatch on first use.

        Completion c;
        c.node_id = s.node_id;
        c.image   = std::move(s.image);
        c.ticket  = s.ticket;
        out.push_back(std::move(c));

        s.state   = SlotState::Free;
        s.ticket  = 0;
        s.node_id = 0;
        s.needs_acquire_on_graphics = false;
    }
    return out;
}

} // namespace te