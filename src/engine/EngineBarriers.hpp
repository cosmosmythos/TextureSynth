#pragma once
#include <vulkan/vulkan.h>

namespace te {

// Generic image layout transition.
void transition(VkCommandBuffer cmd, VkImage img,
                VkImageLayout old_layout, VkImageLayout new_layout,
                VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access);

// Compute shader storage write -> compute shader sampled read (GENERAL->GENERAL).
void barrier_compute_write_to_sampled(VkCommandBuffer cmd, VkImage img);

// Compute shader storage write -> compute shader sampled+storage read (GENERAL->GENERAL).
void barrier_compute_write_to_sampled_read(VkCommandBuffer cmd, VkImage img);

// Transition an output image to GENERAL, using UNDEFINED as old layout for aliased resources.
void transition_output_to_general(VkCommandBuffer cmd, VkImage img,
                                  VkImageLayout current_layout,
                                  uint32_t alias_group_id);

} // namespace te
