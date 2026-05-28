#define VMA_IMPLEMENTATION
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
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

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.descriptorIndexing                                = VK_TRUE;
    f12.shaderSampledImageArrayNonUniformIndexing         = VK_TRUE;
    f12.shaderStorageImageArrayNonUniformIndexing         = VK_TRUE;
    f12.descriptorBindingSampledImageUpdateAfterBind      = VK_TRUE;
    f12.descriptorBindingStorageImageUpdateAfterBind      = VK_TRUE;
    f12.descriptorBindingUpdateUnusedWhilePending         = VK_TRUE;
    f12.descriptorBindingPartiallyBound                   = VK_TRUE;
    f12.runtimeDescriptorArray                            = VK_TRUE;
    f12.bufferDeviceAddress                               = VK_TRUE;

    vkb::PhysicalDeviceSelector ps{vkb_instance_};
    ps.set_minimum_version(1, 3)
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

    // ── Graphics queue ─────────────────────────────────────────────
    auto gq  = vkb_device_.get_queue(vkb::QueueType::graphics);
    auto gqf = vkb_device_.get_queue_index(vkb::QueueType::graphics);
    if (!gq || !gqf) { log_error("no graphics queue"); return false; }
    graphics_queue_  = gq.value();
    graphics_family_ = gqf.value();

    // ── Dedicated transfer queue (best effort) ─────────────────────
    // vkb::QueueType::transfer prefers a family that has TRANSFER but
    // NOT graphics/compute (DMA engine on discrete GPUs).
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

    // ── VMA ────────────────────────────────────────────────────────
    VmaAllocatorCreateInfo aci{};
    aci.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    aci.physicalDevice   = physical_device_;
    aci.device           = device_;
    aci.instance         = instance_;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    if (vmaCreateAllocator(&aci, &allocator_) != VK_SUCCESS) {
        log_error("VMA init failed"); return false;
    }

    // ── Pipeline cache (P6) ────────────────────────────────────────
    VkPipelineCacheCreateInfo pcci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    if (vkCreatePipelineCache(device_, &pcci, nullptr, &pipeline_cache_) != VK_SUCCESS) {
        log_warn("VkPipelineCache creation failed; pipelines will recompile cold");
        pipeline_cache_ = VK_NULL_HANDLE;
    }

    log_info("Vulkan context ready");
    return true;
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