#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <VkBootstrap.h>
#include <mutex>
#include <optional>
#include <string>

namespace te {

struct VulkanContextDesc {
    bool enable_validation = true;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    const char** extra_instance_extensions = nullptr;
    uint32_t extra_instance_extension_count = 0;
};

class VulkanContext {
public:
    bool init(const VulkanContextDesc& desc);
    void shutdown();

    VkInstance        instance()        const { return instance_; }
    VkPhysicalDevice  physical_device() const { return physical_device_; }
    VkDevice          device()          const { return device_; }

    VkQueue           graphics_queue()  const { return graphics_queue_; }
    uint32_t          graphics_family() const { return graphics_family_; }

    // Transfer queue: distinct family if available, otherwise == graphics_queue. Always non-null after init().
    VkQueue           transfer_queue()  const { return transfer_queue_; }
    uint32_t          transfer_family() const { return transfer_family_; }
    bool              has_dedicated_transfer() const { return transfer_queue_ != graphics_queue_; }

    VmaAllocator      allocator()       const { return allocator_; }

    // Pipeline cache shared across all ComputePipeline::create() calls.
    VkPipelineCache   pipeline_cache()  const { return pipeline_cache_; }

    // GPU timestamp period (nanoseconds per timestamp tick). Read from VkPhysicalDeviceLimits at init.
    double            timestamp_period() const { return timestamp_period_; }

    // Thread-safe queue submission. Hold for duration of vkQueueSubmit (NOT thread-safe per spec).
    std::mutex& graphics_queue_mutex() { return graphics_queue_mu_; }
    std::mutex& transfer_queue_mutex() {
        return has_dedicated_transfer() ? transfer_queue_mu_ : graphics_queue_mu_;
    }

    // Serialize the pipeline cache to a blob (call before shutdown).
    bool save_pipeline_cache(const std::string& path) const;
    bool load_pipeline_cache(const std::string& path);

    bool format_supports_image_usage(VkFormat fmt,
                                     VkImageUsageFlags usage,
                                     std::string* error = nullptr) const;

    // Debug object naming (VK_EXT_debug_utils). No-op if extension unavailable. Name must outlive call (VVL does not copy it - VVL#1168). Returns VK_SUCCESS for no-ops so callers can chain without if-guard. Names >255 chars truncated to Vulkan spec limit.
    VkResult set_debug_name(VkObjectType type, uint64_t handle,
                            const std::string& name) const;

    vkb::Instance&        bootstrap_instance()  { return vkb_instance_; }
    vkb::PhysicalDevice&  bootstrap_phys()      { return vkb_phys_; }
    vkb::Device&          bootstrap_device()    { return vkb_device_; }

private:
    vkb::Instance        vkb_instance_{};
    vkb::PhysicalDevice  vkb_phys_{};
    vkb::Device          vkb_device_{};

    VkInstance       instance_        = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice         device_          = VK_NULL_HANDLE;

    VkQueue          graphics_queue_  = VK_NULL_HANDLE;
    uint32_t         graphics_family_ = 0;
    VkQueue          transfer_queue_  = VK_NULL_HANDLE;
    uint32_t         transfer_family_ = 0;

    VmaAllocator     allocator_       = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkPipelineCache  pipeline_cache_  = VK_NULL_HANDLE;
    double           timestamp_period_ = 1.0;

    // PFN_vkSetDebugUtilsObjectNameEXT, loaded at init() if available. Cached so per-object naming is one call, not GetDeviceProcAddr per object.
    PFN_vkSetDebugUtilsObjectNameEXT set_debug_name_fn_ = nullptr;

    std::mutex       graphics_queue_mu_;
    std::mutex       transfer_queue_mu_;
};

} // namespace te
