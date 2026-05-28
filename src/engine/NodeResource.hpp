#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "engine/Graph.hpp"
#include <string>

namespace te {

struct NodeResource {
    NodeId        node_id      = 0;
    int           output_index = 0;
    VkImage       image        = VK_NULL_HANDLE;
    VmaAllocation alloc        = nullptr;
    VkImageView   view         = VK_NULL_HANDLE;
    VkExtent3D    extent       = {0, 0, 1};
    VkFormat      format       = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkImageLayout layout       = VK_IMAGE_LAYOUT_UNDEFINED;
    std::string   debug_name;
    bool          is_dirty     = true;
    uint64_t      last_evaluated_revision = 0;
};

} // namespace te