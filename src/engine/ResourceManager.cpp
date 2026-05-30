#include "engine/ResourceManager.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/Graph.hpp"
#include <algorithm>


namespace te {


constexpr uint32_t RETIRE_FRAMES = 4; // MAX_FRAMES_IN_FLIGHT + safety


static VkFormat resolve_node_format(ChannelFormat override_format,
                                    const NodeLibrary& lib,
                                    const std::string& type_id,
                                    VkFormat default_format) {
	if (override_format != ChannelFormat::RGBA) {
		return channel_to_vk_format(override_format);
	}
	auto* type = lib.find(type_id);
	if (type && !type->outputs.empty()) {
		return channel_to_vk_format(type->outputs[0].format);
	}
	return default_format;
}


size_t ResourceManager::pixel_bytes_(VkFormat f) const {
    switch (f) {
        // 32-bit
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        case VK_FORMAT_R32G32B32_SFLOAT:    return 12;
        case VK_FORMAT_R32G32_SFLOAT:       return 8;
        case VK_FORMAT_R32_SFLOAT:          return 4;
        case VK_FORMAT_R32_UINT:            return 4;
        // 16-bit
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R16G16B16_SFLOAT:    return 6;
        case VK_FORMAT_R16G16_SFLOAT:       return 4;
        case VK_FORMAT_R16_SFLOAT:          return 2;
        case VK_FORMAT_R16_UINT:            return 2;
        // 8-bit
        case VK_FORMAT_R8G8B8A8_UNORM:      return 4;
        case VK_FORMAT_R8G8B8_UNORM:        return 3;
        case VK_FORMAT_R8G8_UNORM:          return 2;
        case VK_FORMAT_R8_UNORM:            return 1;
        case VK_FORMAT_R8_UINT:             return 1;
        default: return 4;
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
                                         const GraphIR& ir, const NodeLibrary& lib,
                                         uint32_t width, uint32_t height,
                                         VkFormat default_format,
                                         std::string* error) {
    retire_all(ctx); // previous graph's images become retired

    size_t total = 0;
    for (const auto& vn : ir.nodes) {
        const auto* type = lib.find(vn.type_id);
        const uint32_t num_outputs = type ? std::max(1u, (uint32_t)type->outputs.size()) : 1;
        for (uint32_t output_id = 0; output_id < num_outputs; ++output_id) {
            VkFormat format_ = resolve_node_format(vn.format_override, lib, vn.type_id, default_format);
            total += (size_t)width * height * pixel_bytes_(format_);
        }
    }

    if (total > budget_bytes_) {
        std::string e = "Memory Budget Exceeded: Need "
                        + std::to_string(total / (1024 * 1024)) + " MB, budget "
                        + std::to_string(budget_bytes_ / (1024 * 1024)) + " MB";
        if (error) *error = e;
        log_error(e);
        return false;
    }

    for (const auto& vn : ir.nodes) {
        const auto& type = lib.find(vn.type_id);
        if (!type) continue;
        const uint32_t num_outputs = std::max(1u, (uint32_t)type->outputs.size());
        for (uint32_t output_id = 0; output_id < num_outputs; ++output_id) {
            NodeResource r;
            r.node_id = vn.id;
            r.output_index = output_id;
            VkFormat format_ = default_format;
            if (output_id < type->outputs.size()) {
                format_ = resolve_node_format(vn.format_override, lib, vn.type_id, default_format);
                if (type->outputs[output_id].format != ChannelFormat::RGBA) {
                    format_ = channel_to_vk_format(type->outputs[output_id].format);
                }
            }
            r.format = format_;
            if (!create_image_(ctx, r, width, height, vn.debug_name + "_out" + std::to_string(output_id))) {
                if (error) *error = "image allocation failed for " + vn.debug_name;
                return false;
            }
            live_[{vn.id, output_id}] = std::move(r);
            current_bytes_ += (size_t)width * height * pixel_bytes_(format_);
        }
    }
    log_info("ResourceManager: Allocated " + std::to_string(live_.size())
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