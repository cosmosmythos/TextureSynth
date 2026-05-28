#include "viewer/Swapchain.hpp"
#include <iostream>

namespace te {

bool Swapchain::init(VulkanContext& ctx, VkSurfaceKHR surface, uint32_t width, uint32_t height) {
    ctx_ = &ctx;
    surface_ = surface;
    return recreate(width, height);
}

void Swapchain::shutdown() {
    cleanup_old();
}

void Swapchain::cleanup_old() {
    if (swapchain_ != VK_NULL_HANDLE) {
        for (auto view : image_views_) {
            vkDestroyImageView(ctx_->device(), view, nullptr);
        }
        image_views_.clear();
        images_.clear();
        
        vkb::destroy_swapchain(vkb_swapchain_);
        swapchain_ = VK_NULL_HANDLE;
    }
}

bool Swapchain::recreate(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return true; // Minimized
    
    vkb::SwapchainBuilder swapchain_builder{ctx_->bootstrap_device(), surface_};
    
    auto vkb_swap_ret = swapchain_builder
        .set_desired_extent(width, height)
        .set_old_swapchain(swapchain_)
        .add_image_usage_flags(VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        .build();

    if (!vkb_swap_ret) {
        std::cerr << "Failed to create swapchain: " << vkb_swap_ret.error().message() << "\n";
        return false;
    }

    cleanup_old();

    vkb_swapchain_ = vkb_swap_ret.value();
    swapchain_ = vkb_swapchain_.swapchain;
    image_format_ = vkb_swapchain_.image_format;
    extent_ = vkb_swapchain_.extent;

    auto images_ret = vkb_swapchain_.get_images();
    if (images_ret) {
        images_ = images_ret.value();
    }
    
    auto views_ret = vkb_swapchain_.get_image_views();
    if (views_ret) {
        image_views_ = views_ret.value();
    }

    return true;
}

} // namespace te
