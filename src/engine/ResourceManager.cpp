#include "engine/ResourceManager.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
#include <algorithm>

namespace te {

constexpr uint32_t RETIRE_FRAMES = 4; // MAX_FRAMES_IN_FLIGHT + safety

size_t ResourceManager::pixel_bytes_(VkFormat f) const {
    switch (f) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R8G8B8A8_UNORM:      return 4;
        default: return 16;
    }
}

bool ResourceManager::create_image_(VulkanContext& ctx, NodeResource& r,
                                    uint32_t w, uint32_t h,
                                    const std::string& dbg) {
    r.extent = {w, h, 1};
    r.debug_name = dbg;
    r.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format    = r.format;
    ici.extent    = r.extent;
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples   = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
    ici.usage     = VK_IMAGE_USAGE_STORAGE_BIT
                  | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                  | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(ctx.allocator(), &ici, &aci, &r.image, &r.alloc, nullptr) != VK_SUCCESS) {
        log_error("ResourceManager: vmaCreateImage failed for " + dbg);
        return false;
    }

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image    = r.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = r.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device(), &vci, nullptr, &r.view) != VK_SUCCESS) {
        log_error("ResourceManager: view creation failed for " + dbg);
        vmaDestroyImage(ctx.allocator(), r.image, r.alloc);
        r.image = VK_NULL_HANDLE;
        r.alloc = nullptr;
        return false;
    }
    return true;
}

void ResourceManager::destroy_image_(VulkanContext& ctx, NodeResource& r) {
    if (r.view)  { vkDestroyImageView(ctx.device(), r.view, nullptr); r.view = VK_NULL_HANDLE; }
    if (r.image) { vmaDestroyImage(ctx.allocator(), r.image, r.alloc);
                   r.image = VK_NULL_HANDLE; r.alloc = nullptr; }
}

bool ResourceManager::allocate_for_graph(VulkanContext& ctx,
                                         const GraphIR& ir,
                                         uint32_t width, uint32_t height,
                                         VkFormat format,
                                         std::string* error) {
    retire_all(ctx); // previous graph's images become retired

    const size_t per_image = (size_t)width * height * pixel_bytes_(format);
    const size_t total = per_image * ir.nodes.size();
    if (total > budget_bytes_) {
        std::string e = "memory budget exceeded: need "
                        + std::to_string(total / (1024 * 1024)) + " MB, budget "
                        + std::to_string(budget_bytes_ / (1024 * 1024)) + " MB";
        if (error) *error = e;
        log_error(e);
        return false;
    }

    for (const auto& vn : ir.nodes) {
        NodeResource r;
        r.node_id      = vn.id;
        r.output_index = 0;
        r.format       = format;
        if (!create_image_(ctx, r, width, height, vn.debug_name)) {
            if (error) *error = "image allocation failed for " + vn.debug_name;
            return false;
        }
        live_[{vn.id, 0}] = std::move(r);
        current_bytes_ += per_image;
    }
    log_info("ResourceManager: allocated " + std::to_string(live_.size())
             + " images, " + std::to_string(current_bytes_ / (1024 * 1024)) + " MB");
    return true;
}

void ResourceManager::retire_all(VulkanContext& ctx) {
    for (auto& kv : live_) {
        retired_.push_back({std::move(kv.second), RETIRE_FRAMES});
    }
    live_.clear();
    current_bytes_ = 0;
    (void)ctx;
}

void ResourceManager::tick(VulkanContext& ctx) {
    for (auto& r : retired_) if (r.frames_remaining > 0) --r.frames_remaining;
    retired_.erase(
        std::remove_if(retired_.begin(), retired_.end(),
            [&](Retired& r) {
                if (r.frames_remaining == 0) {
                    destroy_image_(ctx, r.res);
                    return true;
                }
                return false;
            }),
        retired_.end());
}

void ResourceManager::shutdown(VulkanContext& ctx) {
    for (auto& kv : live_)    destroy_image_(ctx, kv.second);
    for (auto& r  : retired_) destroy_image_(ctx, r.res);
    live_.clear();
    retired_.clear();
    current_bytes_ = 0;
}

} // namespace te