#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include "engine/Image.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace te {

class VulkanContext;

// Async, pooled image uploader. Replaces Image::upload_pixels' synchronous path.
// - One persistent command pool per slot (transient + resettable).
// - One staging buffer per slot, grown on demand.
// - Fence-tracked: poll() returns finished slots without blocking.
// - Uses ctx.transfer_queue() and acquires its mutex for submission.
class ImageUploader {
public:
    static constexpr uint32_t SLOT_COUNT = 4;

    struct Completion {
        uint64_t              node_id  = 0;
        std::unique_ptr<Image> image;          // ownership transferred to caller
        uint64_t              ticket   = 0;
    };

    bool init(VulkanContext& ctx);
    void shutdown(VulkanContext& ctx);

    // Submit a non-blocking upload. Returns ticket > 0 on success,
    // 0 if the ring is full (caller retries next tick).
    // `out_image` will be created/recreated to match (w,h) and populated
    // by the GPU. Caller may NOT touch out_image until poll() returns it.
    uint64_t submit(VulkanContext& ctx,
                    uint64_t node_id,
                    const float* pixels, uint32_t w, uint32_t h,
                    std::unique_ptr<Image> recycled_image = nullptr);

    // Drain finished slots; caller takes ownership of completed Image*.
    std::vector<Completion> poll(VulkanContext& ctx);

    // Block until all in-flight slots finish (shutdown / topology change).
    void drain(VulkanContext& ctx);

    bool any_in_flight() const;

private:
    enum class SlotState { Free, InFlight };

    struct Slot {
        VkCommandPool   pool   = VK_NULL_HANDLE;
        VkCommandBuffer cmd    = VK_NULL_HANDLE;
        VkFence         fence  = VK_NULL_HANDLE;

        VkBuffer        staging          = VK_NULL_HANDLE;
        VmaAllocation   staging_alloc    = nullptr;
        void*           staging_mapped   = nullptr;
        VkDeviceSize    staging_capacity = 0;

        // The in-flight job's payload — moves to Completion on poll().
        std::unique_ptr<Image> image;
        uint64_t node_id = 0;
        uint64_t ticket  = 0;

        // Bookkeeping for the QFOT acquire on the graphics queue.
        // If transfer family != graphics family, we issued a release
        // barrier on transfer and need a matching acquire later.
        bool needs_acquire_on_graphics = false;
        uint32_t job_w = 0, job_h = 0;

        SlotState state = SlotState::Free;
    };

    bool init_slot_(VulkanContext& ctx, Slot& s);
    void destroy_slot_(VulkanContext& ctx, Slot& s);
    bool grow_staging_(VulkanContext& ctx, Slot& s, VkDeviceSize cap);

    std::array<Slot, SLOT_COUNT> slots_;
    uint64_t next_ticket_ = 1;
};

} // namespace te