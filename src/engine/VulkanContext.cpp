#define VMA_IMPLEMENTATION
#include "engine/VulkanContext.hpp"
#include "engine/Graph.hpp"
#include "engine/Logging.hpp"
#include <algorithm>
#include <fstream>
#include <vector>

namespace te {

bool VulkanContext::init(const VulkanContextDesc& desc) {
    vkb::InstanceBuilder ib;
    ib.set_app_name("texture_engine")
      .require_api_version(1, 3, 0);
    if (desc.enable_validation) {
        ib.request_validation_layers(true)
          .use_default_debug_messenger();
    }
    for (uint32_t i = 0; i < desc.extra_instance_extension_count; ++i) {
        ib.enable_extension(desc.extra_instance_extensions[i]);
    }
    auto inst_ret = ib.build();
    if (!inst_ret) { log_error(inst_ret.error().message()); return false; }
    vkb_instance_    = inst_ret.value();
    instance_        = vkb_instance_.instance;
    debug_messenger_ = vkb_instance_.debug_messenger;

    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.synchronization2 = VK_TRUE;
    f13.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12 {};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.descriptorIndexing                                = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing         = VK_TRUE;
    f12.shaderStorageImageArrayNonUniformIndexing         = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind      = VK_TRUE;
    f12.descriptorBindingStorageImageUpdateAfterBind      = VK_TRUE;
    f12.descriptorBindingStorageBufferUpdateAfterBind     = VK_TRUE;
    f12.descriptorBindingUpdateUnusedWhilePending         = VK_TRUE;
    f12.descriptorBindingPartiallyBound                   = VK_TRUE;
    f12.runtimeDescriptorArray                            = VK_TRUE;
    f12.bufferDeviceAddress                               = VK_TRUE;

    VkPhysicalDeviceFeatures f10{};
    f10.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    f10.shaderStorageImageExtendedFormats    = VK_TRUE;

    vkb::PhysicalDeviceSelector ps{vkb_instance_};
    ps.set_minimum_version(1, 3)
      .set_required_features(f10)
      .set_required_features_13(f13)
      .set_required_features_12(f12);
    if (desc.surface != VK_NULL_HANDLE) ps.set_surface(desc.surface);
    else                                ps.defer_surface_initialization();

    auto phys_ret = ps.select();
    if (!phys_ret) { log_error(phys_ret.error().message()); return false; }
    vkb_phys_        = phys_ret.value();
    physical_device_ = vkb_phys_.physical_device;

    vkb::DeviceBuilder db{vkb_phys_};
    auto dev_ret = db.build();
    if (!dev_ret) { log_error(dev_ret.error().message()); return false; }
    vkb_device_ = dev_ret.value();
    device_     = vkb_device_.device;

    const VkImageUsageFlags node_image_usage =
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    bool ok = true;
    std::vector<VkFormat> checked;
    size_t storage_count = 0;
    const StorageFormatInfo* storage_infos = storage_format_info_table(storage_count);
    for (size_t i = 0; i < storage_count; ++i) {
        const VkFormat fmt = storage_infos[i].vk_format;
        if (std::find(checked.begin(), checked.end(), fmt) != checked.end()) continue;
        checked.push_back(fmt);
        ok &= format_supports_image_usage(fmt, node_image_usage);
    }
    if (!ok) { log_error("Required image formats not supported"); return false; }

    // GPU timestamp period (nanoseconds per tick).
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    timestamp_period_ = props.limits.timestampPeriod;
    if (!props.limits.timestampComputeAndGraphics) {
        log_warn("VulkanContext: timestampComputeAndGraphics is false — "
                 "timestamps may return zero on compute queue");
    }

    // Graphics queue
    auto gq  = vkb_device_.get_queue(vkb::QueueType::graphics);
    auto gqf = vkb_device_.get_queue_index(vkb::QueueType::graphics);
    if (!gq || !gqf) { log_error("no graphics queue"); return false; }
    graphics_queue_  = gq.value();
    graphics_family_ = gqf.value();

    // Dedicated transfer queue (best effort). vkb::QueueType::transfer prefers a family with TRANSFER but NOT graphics/compute (DMA engine on discrete GPUs).
    auto tq  = vkb_device_.get_dedicated_queue(vkb::QueueType::transfer);
    auto tqf = vkb_device_.get_dedicated_queue_index(vkb::QueueType::transfer);
    if (tq && tqf && tqf.value() != graphics_family_) {
        transfer_queue_  = tq.value();
        transfer_family_ = tqf.value();
        log_info("VulkanContext: dedicated transfer queue family="
                 + std::to_string(transfer_family_));
    } else {
        transfer_queue_  = graphics_queue_;
        transfer_family_ = graphics_family_;
        log_info("VulkanContext: using graphics queue for transfers");
    }

    // VMA
    VmaAllocatorCreateInfo aci{};
    // BUFFER_DEVICE_ADDRESS_BIT: needed for indirect-address descriptor access. EXT_MEMORY_BUDGET_BIT: without it vmaGetHeapBudgets returns static estimate instead of real OS-level residency budget.
    aci.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
                         | VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    aci.physicalDevice   = physical_device_;
    aci.device           = device_;
    aci.instance         = instance_;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    if (vmaCreateAllocator(&aci, &allocator_) != VK_SUCCESS) {
        log_error("VMA init failed"); return false;
    }

    // Pipeline cache
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (vkCreatePipelineCache(device_, &pcci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        log_warn("VkPipelineCache creation failed; pipelines will recompile cold");
        pipeline_cache_ = VK_NULL_HANDLE;
    }

    // VK_EXT_debug_utils: vkSetDebugUtilsObjectNameEXT. Core in Vulkan 1.1+ but loaded by pointer for portability.
    set_debug_name_fn_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(device_, "vkSetDebugUtilsObjectNameEXT"));
    if (set_debug_name_fn_) {
        log_info("VulkanContext: vkSetDebugUtilsObjectNameEXT loaded");
    } else {
        log_info("VulkanContext: vkSetDebugUtilsObjectNameEXT not available "
                 "(debug object names will be no-ops)");
    }

    log_info("Vulkan context ready");
    return true;
}

bool VulkanContext::format_supports_image_usage(VkFormat fmt,
                                                VkImageUsageFlags usage,
                                                std::string* error) const {
    VkFormatFeatureFlags required = 0;
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
        required |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
        required |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    if (usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
        required |= VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    if (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
        required |= VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(physical_device_, fmt, &props);
    const VkFormatFeatureFlags available = props.optimalTilingFeatures;
    if ((available & required) == required) return true;

    std::string msg = std::string("Format ") + vk_format_name(fmt)
                    + " lacks required optimal-tiling image features";
    if (error) *error = msg;
    log_error(msg);
    return false;
}

bool VulkanContext::save_pipeline_cache(const std::string& path) const {
    if (!pipeline_cache_ || !device_) return false;
    size_t sz = 0;
    if (vkGetPipelineCacheData(device_, pipeline_cache_, &sz, nullptr) != VK_SUCCESS || sz == 0)
        return false;
    std::vector<uint8_t> blob(sz);
    if (vkGetPipelineCacheData(device_, pipeline_cache_, &sz, blob.data()) != VK_SUCCESS)
        return false;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(blob.data()), (std::streamsize)sz);
    return true;
}

bool VulkanContext::load_pipeline_cache(const std::string& path) {
    if (!device_) return false;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto sz = (size_t)f.tellg();
    if (sz == 0) return false;
    f.seekg(0);
    std::vector<uint8_t> blob(sz);
    f.read(reinterpret_cast<char*>(blob.data()), (std::streamsize)sz);

    // Replace any cache created in init() with one seeded from disk.
    if (pipeline_cache_) {
        vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
        pipeline_cache_ = VK_NULL_HANDLE;
    }
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pcci.initialDataSize = sz;
    pcci.pInitialData    = blob.data();
    if (vkCreatePipelineCache(device_, &pcci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        log_warn("pipeline cache load failed; starting cold");
        VkPipelineCacheCreateInfo empty{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        vkCreatePipelineCache(device_, &empty, nullptr, &pipeline_cache_);
        return false;
    }
    log_info("pipeline cache loaded (" + std::to_string(sz) + " bytes)");
    return true;
}

VkResult VulkanContext::set_debug_name(VkObjectType type, uint64_t handle,
                                       const std::string& name) const {
    // No-op fast paths. Return VK_SUCCESS so callers can chain without if-guard.
    if (!set_debug_name_fn_ || handle == 0 || name.empty()) {
        return VK_SUCCESS;
    }
    // Vulkan spec: pObjectName max 256 bytes including NUL. Hard-code 256; VVL rejects longer strings. Truncate proactively.
    constexpr size_t kMaxUsable = 256 - 1;  // 255 chars
    std::string trimmed = name;
    if (trimmed.size() > kMaxUsable) {
        log_warn("set_debug_name: truncating name from " + std::to_string(trimmed.size())
                 + " to " + std::to_string(kMaxUsable) + " chars: \""
                 + trimmed.substr(0, 32) + "...\"");
        trimmed.resize(kMaxUsable);
    }
    // VVL does not copy pObjectName (VVL#1168); std::string guarantees bytes are live for call duration.
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType   = type;
    info.objectHandle = handle;
    info.pObjectName  = trimmed.c_str();
    return set_debug_name_fn_(device_, &info);
}


void VulkanContext::shutdown() {
    if (pipeline_cache_) {
        vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
        pipeline_cache_ = VK_NULL_HANDLE;
    }
    if (allocator_) { vmaDestroyAllocator(allocator_); allocator_ = nullptr; }
    if (device_)    { vkb::destroy_device(vkb_device_); device_ = VK_NULL_HANDLE; }
    if (instance_)  { vkb::destroy_instance(vkb_instance_); instance_ = VK_NULL_HANDLE; }
}

} // namespace te
