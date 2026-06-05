#pragma once
#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace te {

class VulkanContext;

// One global, persistent descriptor set: binding 0 = sampled_images[MAX_SAMPLED] (UPDATE_AFTER_BIND, PARTIALLY_BOUND), binding 1 = storage_images[MAX_STORAGE], bindings 2-4 = immutable samplers (repeat, clamp, mirror). One pipeline layout (push constants only). Bound once; never rebound.
class BindlessTable {
public:
    static constexpr uint32_t MAX_SAMPLED = 4096;
    static constexpr uint32_t MAX_STORAGE = 4096;
    static constexpr uint32_t INVALID_SLOT = 0xFFFFFFFFu;
    static constexpr uint32_t PARAM_RING_SIZE = 3;  // MAX_FRAMES_IN_FLIGHT + 1

    bool init(VulkanContext& ctx,
              VkSampler samp_repeat,
              VkSampler samp_clamp,
              VkSampler samp_mirror,
              uint32_t push_constant_size);
    void shutdown(VulkanContext& ctx);

    // Slot allocation (free-list LIFO; O(1)).
    uint32_t alloc_sampled_slot();
    uint32_t alloc_storage_slot();
    void     free_sampled_slot(uint32_t slot);
    void     free_storage_slot(uint32_t slot);

    // Descriptor writes
    void write_sampled(VulkanContext& ctx, uint32_t slot, VkImageView view, VkImageLayout layout);
    void write_storage(VulkanContext& ctx, uint32_t slot, VkImageView view);
    void write_param_ring(VulkanContext& ctx, const std::array<VkBuffer, PARAM_RING_SIZE>& bufs, VkDeviceSize range);

    VkDescriptorSetLayout set_layout()    const { return set_layout_; }
    VkPipelineLayout      pipeline_layout() const { return pipe_layout_; }
    VkDescriptorSet       set()           const { return set_; }

private:
    VkDescriptorPool      pool_        = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_  = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout_ = VK_NULL_HANDLE;
    VkDescriptorSet       set_         = VK_NULL_HANDLE;

    std::vector<uint32_t> sampled_free_;
    std::vector<uint32_t> storage_free_;
    uint32_t sampled_next_ = 0;
    uint32_t storage_next_ = 0;
};

} // namespace te