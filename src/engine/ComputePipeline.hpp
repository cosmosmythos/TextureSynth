#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace te {

class VulkanContext;

class ComputePipeline {
public:
    // input_image_count = number of in_tex_N bindings (bindings 2..2+N-1).
    // spec: optional VkSpecializationInfo. Pass nullptr (the default) for
    //       un-specialized pipelines. When non-null, the data must remain
    //       valid for the duration of this call only (vkCreateComputePipelines
    //       reads it synchronously per the Vulkan spec).
    bool create(VulkanContext& ctx,
                const std::vector<uint32_t>& spirv,
                VkPipelineLayout layout,
                const VkSpecializationInfo* spec = nullptr);
    void destroy(VulkanContext& ctx);

    // Stable storage for the VkDebugUtilsObjectNameEXT name. Owned here so
    // the string outlives the validation layer's pointer (see Khronos
    // VVL#1168 - VVL does not copy pObjectName).
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