#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdint>
#include <array>
#include <vector>

namespace te {
class VulkanContext;
class Engine;
struct PushConstants;

class AsyncReadback {
public:
    static constexpr uint32_t SLOT_COUNT = 3;

    bool init(VulkanContext& ctx, uint32_t max_width, uint32_t max_height);
    void shutdown(VulkanContext& ctx);
    bool ensure_capacity(VulkanContext& ctx, uint32_t w, uint32_t h);

    uint64_t submit(VulkanContext& ctx,
                    Engine& engine,
                    const PushConstants& pc,
                    uint64_t generation);

    bool poll(VulkanContext& ctx,
              std::vector<float>& out_pixels,
              uint32_t& out_w, uint32_t& out_h,
              uint64_t& out_generation);

    bool any_in_flight() const;

    // Block until every in-flight slot has finished. Required before
    // any topology change that would invalidate descriptors/resources.
    void drain(VulkanContext& ctx);

    // Publish pre-cached pixels without GPU work (for dirty-skip path).
    uint64_t publish_synthetic(const std::vector<float>& pixels,
                               uint32_t w, uint32_t h,
                               uint64_t generation);

private:
    enum class SlotState { Free, InFlight };
    struct Slot {
        VkCommandPool   pool   = VK_NULL_HANDLE;
        VkCommandBuffer cmd    = VK_NULL_HANDLE;
        VkFence         fence  = VK_NULL_HANDLE;
        VkBuffer        staging        = VK_NULL_HANDLE;
        VmaAllocation   staging_alloc  = nullptr;
        void*           staging_mapped = nullptr;
        VkDeviceSize    staging_capacity = 0;
        SlotState state = SlotState::Free;
        uint32_t  job_w = 0;
        uint32_t  job_h = 0;
        uint64_t  job_generation = 0;
        uint64_t  ticket = 0;
    };

    bool init_slot_(VulkanContext& ctx, Slot& s, VkDeviceSize capacity);
    void destroy_slot_(VulkanContext& ctx, Slot& s);
    bool grow_staging_(VulkanContext& ctx, Slot& s, VkDeviceSize new_capacity);

    std::array<Slot, SLOT_COUNT> slots_;
    uint32_t  current_capacity_w_ = 0;
    uint32_t  current_capacity_h_ = 0;
    uint64_t  next_ticket_ = 1;
    uint64_t  latest_submitted_generation_ = 0;
    // Set true on a successful init(), false on shutdown() and on the
    // default-constructed state. submit() and poll() return early if false,
    // so a use-after-shutdown at the AsyncReadback level is a clean no-op
    // rather than a use-after-free on a null VkBuffer.
    bool      initialized_ = false;

    // Synthetic (CPU-cached) frame — populated by publish_synthetic().
    std::vector<float> synthetic_pixels_;
    uint32_t           synthetic_w_ = 0;
    uint32_t           synthetic_h_ = 0;
    uint64_t           synthetic_generation_ = 0;
    uint64_t           synthetic_ticket_ = 0;
    bool               synthetic_ready_ = false;
};
} // namespace te
