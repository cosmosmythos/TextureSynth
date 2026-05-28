#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace te {

class VulkanContext;

class ComputePipeline {
public:
    // input_image_count = number of in_tex_N bindings (bindings 2..2+N-1).
    bool create(VulkanContext& ctx,
                const std::vector<uint32_t>& spirv,
                VkPipelineLayout layout);
    void destroy(VulkanContext& ctx);

    VkPipeline            pipeline()    const { return pipeline_; }
    VkPipelineLayout      layout()      const { return layout_; }

private:
    VkShaderModule        module_      = VK_NULL_HANDLE;
    VkPipelineLayout      layout_      = VK_NULL_HANDLE;
    VkPipeline            pipeline_    = VK_NULL_HANDLE;
};

} // namespace te