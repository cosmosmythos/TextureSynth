#pragma once
#include "engine/VulkanContext.hpp"
#include <vector>

namespace te {

class Swapchain {
public:
    bool init(VulkanContext& ctx, VkSurfaceKHR surface, uint32_t width, uint32_t height);
    void shutdown();
    bool recreate(uint32_t width, uint32_t height);

    VkSwapchainKHR handle() const { return swapchain_; }
    VkFormat image_format() const { return image_format_; }
    VkExtent2D extent() const { return extent_; }
    const std::vector<VkImage>& images() const { return images_; }
    const std::vector<VkImageView>& image_views() const { return image_views_; }

private:
    void cleanup_old();

    VulkanContext* ctx_ = nullptr;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    vkb::Swapchain vkb_swapchain_{};
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkFormat image_format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{0, 0};
    
    std::vector<VkImage> images_;
    std::vector<VkImageView> image_views_;
};

} // namespace te
