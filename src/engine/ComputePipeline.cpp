#include "engine/ComputePipeline.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
#include <vector>

namespace te {

bool ComputePipeline::create(VulkanContext& ctx,
                             const std::vector<uint32_t>& spirv,
                             VkPipelineLayout layout,
                             const VkSpecializationInfo* spec) {
    if (spirv.empty()) { log_error("ComputePipeline: empty SPIR-V"); return false; }
    layout_ = layout;

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode    = spirv.data();
    if (vkCreateShaderModule(ctx.device(), &smci, nullptr, &module_) != VK_SUCCESS) {
        log_error("vkCreateShaderModule failed"); return false;
    }

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module_;
    stage.pName  = "main";
    // spec is nullptr for un-specialized pipelines. pSpecializationInfo is read synchronously by vkCreateComputePipelines.
    stage.pSpecializationInfo = spec;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage  = stage;
    cpci.layout = layout_;
    if (vkCreateComputePipelines(ctx.device(), ctx.pipeline_cache(),
                                 1, &cpci, nullptr, &pipeline_) != VK_SUCCESS) {
        log_error("vkCreateComputePipelines failed"); return false;
    }

    vkDestroyShaderModule(ctx.device(), module_, nullptr);
    module_ = VK_NULL_HANDLE;
    return true;
}


void ComputePipeline::destroy(VulkanContext& ctx) {
    if (pipeline_) { vkDestroyPipeline(ctx.device(), pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE; }
    if (module_)   { vkDestroyShaderModule(ctx.device(), module_, nullptr); module_ = VK_NULL_HANDLE; }
    layout_ = VK_NULL_HANDLE;
}

} // namespace te