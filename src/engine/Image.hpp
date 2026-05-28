#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdint>

namespace te {

class VulkanContext;

class Image {
public:
    bool create(VulkanContext& ctx, uint32_t w, uint32_t h,
                VkFormat fmt = VK_FORMAT_R32G32B32A32_SFLOAT,
                VkImageUsageFlags extra_usage = 0);
    bool upload_pixels(VulkanContext& ctx, const float* data, uint32_t w, uint32_t h);
    void destroy(VulkanContext& ctx);

    VkImage     image()  const { return image_; }
    VkImageView view()   const { return view_;  }
    VkExtent2D  extent() const { return {width_, height_}; }
    VkFormat    format() const { return format_; }

private:
    VkImage       image_   = VK_NULL_HANDLE;
    VkImageView   view_    = VK_NULL_HANDLE;
    VmaAllocation alloc_   = VK_NULL_HANDLE;
    uint32_t      width_   = 0;
    uint32_t      height_  = 0;
    VkFormat      format_  = VK_FORMAT_UNDEFINED;
};

} // namespace te