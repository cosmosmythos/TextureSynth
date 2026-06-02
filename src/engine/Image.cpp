#include "engine/Image.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
#include <mutex>

namespace te {

bool Image::create(VulkanContext& ctx, uint32_t w, uint32_t h, VkFormat fmt,
                   VkImageUsageFlags extra_usage) {
    width_ = w; height_ = h; format_ = fmt;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = fmt;
    ici.extent      = {w, h, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_STORAGE_BIT
                    | VK_IMAGE_USAGE_SAMPLED_BIT
                    | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | extra_usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;

    if (vmaCreateImage(ctx.allocator(), &ici, &aci, &image_, &alloc_, nullptr) != VK_SUCCESS) {
        log_error("vmaCreateImage failed"); return false;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image    = image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = fmt;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device(), &vci, nullptr, &view_) != VK_SUCCESS) {
        log_error("vkCreateImageView failed"); return false;
    }
    return true;
}

void Image::destroy(VulkanContext& ctx) {
    if (view_)  { vkDestroyImageView(ctx.device(), view_, nullptr); view_ = VK_NULL_HANDLE; }
    if (image_) { vmaDestroyImage(ctx.allocator(), image_, alloc_); image_ = VK_NULL_HANDLE; alloc_ = VK_NULL_HANDLE; }
}

bool Image::upload_pixels(VulkanContext& ctx, const float* data, uint32_t w, uint32_t h) {

    log_warn("Image::upload_pixels: synchronous path -- use ImageUploader for production");
    if (width_ != w || height_ != h || image_ == VK_NULL_HANDLE) {
        destroy(ctx);
        if (!create(ctx, w, h, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
            log_error("upload_pixels: failed to recreate image");
            return false;
        }
    }

    VkBufferCreateInfo buf_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buf_info.size  = w * h * 4 * sizeof(float);
    buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    
    VmaAllocationCreateInfo vma_ci{};
    vma_ci.usage = VMA_MEMORY_USAGE_AUTO;
    vma_ci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
    
    VkBuffer staging_buf;
    VmaAllocation staging_alloc;
    VmaAllocationInfo staging_info;
    if (vmaCreateBuffer(ctx.allocator(), &buf_info, &vma_ci,
                        &staging_buf, &staging_alloc, &staging_info) != VK_SUCCESS) {
        log_error("upload_pixels: failed to create upload staging buffer");
        return false;
    }

    std::memcpy(staging_info.pMappedData, data, w * h * 4 * sizeof(float));
    vmaFlushAllocation(ctx.allocator(), staging_alloc, 0, VK_WHOLE_SIZE);

    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = ctx.transfer_family();
    VkCommandPool pool;
    vkCreateCommandPool(ctx.device(), &pool_info, nullptr, &pool);

    VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc_info.commandPool        = pool;
    alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(ctx.device(), &alloc_info, &cmd);

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);

    VkImageMemoryBarrier2 to_dst{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    to_dst.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    to_dst.srcAccessMask = 0;
    to_dst.dstStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_dst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_dst.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    to_dst.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_dst.image         = image_;
    to_dst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers    = &to_dst;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent      = {w, h, 1};
    vkCmdCopyBufferToImage(cmd, staging_buf, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier2 to_general = to_dst;
    to_general.srcStageMask  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    to_general.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    to_general.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    to_general.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    to_general.newLayout     = VK_IMAGE_LAYOUT_GENERAL;
    
    dep.pImageMemoryBarriers = &to_general;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fence_ci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(ctx.device(), &fence_ci, nullptr, &fence);

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    {
        std::lock_guard<std::mutex> lk(ctx.transfer_queue_mutex());
        vkQueueSubmit(ctx.transfer_queue(), 1, &submit, fence);
    }
    vkWaitForFences(ctx.device(), 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(ctx.device(), fence, nullptr);
    vmaDestroyBuffer(ctx.allocator(), staging_buf, staging_alloc);
    vkDestroyCommandPool(ctx.device(), pool, nullptr);

    return true;
}

} // namespace te
