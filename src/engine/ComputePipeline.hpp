#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace te {

class VulkanContext;

class ComputePipeline {
public:
    // spec: optional VkSpecializationInfo. Data must remain valid for this call only (vkCreateComputePipelines reads synchronously).
    bool create(VulkanContext& ctx,
                const std::vector<uint32_t>& spirv,
                VkPipelineLayout layout,
                const VkSpecializationInfo* spec = nullptr);
    void destroy(VulkanContext& ctx);

    // Stable storage for the VkDebugUtilsObjectNameEXT name. Owned here so the string outlives VVL's pointer (VVL#1168).
    void set_name(std::string name) { name_ = std::move(name); }
    const std::string& name() const { return name_; }

    VkPipeline            pipeline()    const { return pipeline_; }
    VkPipelineLayout      layout()      const { return layout_; }

private:
    VkShaderModule        module_      = VK_NULL_HANDLE;
    VkPipelineLayout      layout_      = VK_NULL_HANDLE;
    VkPipeline            pipeline_    = VK_NULL_HANDLE;
    std::string           name_;
};

} // namespace te