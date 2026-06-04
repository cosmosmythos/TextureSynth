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

    // Transfer queue: distinct family if available, otherwise == graphics_queue.
    // Always non-null after init().
    VkQueue           transfer_queue()  const { return transfer_queue_; }
    uint32_t          transfer_family() const { return transfer_family_; }
    bool              has_dedicated_transfer() const { return transfer_queue_ != graphics_queue_; }

    VmaAllocator      allocator()       const { return allocator_; }

    // Pipeline cache shared across all ComputePipeline::create() calls.
    VkPipelineCache   pipeline_cache()  const { return pipeline_cache_; }

    // Thread-safe queue submission. Hold for the duration of vkQueueSubmit.
    // Vulkan spec: vkQueueSubmit is NOT thread-safe; one thread per VkQueue at a time.
    std::mutex& graphics_queue_mutex() { return graphics_queue_mu_; }
    std::mutex& transfer_queue_mutex() {
        return has_dedicated_transfer() ? transfer_queue_mu_ : graphics_queue_mu_;
    }

    // Serialize the pipeline cache to a blob (call before shutdown).
    bool save_pipeline_cache(const std::string& path) const;
    bool load_pipeline_cache(const std::string& path);

    // Debug object naming (VK_EXT_debug_utils). No-op if the extension is
    // not available on this loader/driver. The name string must outlive the
    // call (validation layer does not copy it - see Khronos VVL#1168);
    // pass a std::string in scope.
    // Apply a debug name to a Vulkan object. The name is forwarded to
    // vkSetDebugUtilsObjectNameEXT, which is also intercepted by
    // RenderDoc and any custom layer. Returns the VkResult of the
    // underlying call. Returns VK_SUCCESS for harmless no-ops
    // (null fn pointer / null handle / empty name) so callers can
    // chain it without an if-guard. Long names (>255 chars) are
    // truncated to fit the Vulkan spec limit
    // (VK_MAX_DEBUG_UTILS_OBJECT_NAME_LENGTH_EXT = 256 including NUL).
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

    // PFN_vkSetDebugUtilsObjectNameEXT, loaded at init() if VK_EXT_debug_utils
    // is available. Cached as a member so per-object naming is one call,
    // not a GetDeviceProcAddr per object.
    PFN_vkSetDebugUtilsObjectNameEXT set_debug_name_fn_ = nullptr;

    std::mutex       graphics_queue_mu_;
    std::mutex       transfer_queue_mu_;
};

} // namespace te