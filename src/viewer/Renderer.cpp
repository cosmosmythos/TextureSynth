#include "viewer/Renderer.hpp"
#include <iostream>
#include <algorithm>

namespace te {

bool Renderer::init(VulkanContext& ctx, Swapchain& swapchain, Engine& engine, ImGuiLayer& imgui) {
    ctx_ = &ctx;
    swapchain_ = &swapchain;
    engine_ = &engine;
    imgui_ = &imgui;

    create_command_buffers();
    create_sync_objects();

    return true;
}

void Renderer::shutdown() {
    if (ctx_) {
        vkDeviceWaitIdle(ctx_->device());
        destroy_sync_objects();
        vkDestroyCommandPool(ctx_->device(), command_pool_, nullptr);
        ctx_ = nullptr;
    }
}

void Renderer::create_command_buffers() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = ctx_->graphics_family();

    if (vkCreateCommandPool(ctx_->device(), &pool_info, nullptr, &command_pool_) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool");
    }

    command_buffers_.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = (uint32_t)command_buffers_.size();

    if (vkAllocateCommandBuffers(ctx_->device(), &alloc_info, command_buffers_.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate command buffers");
    }
}

void Renderer::create_sync_objects() {
    uint32_t image_count = (uint32_t)swapchain_->images().size();

    VkSemaphoreCreateInfo sem_info{};
    sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    // Acquire semaphores: N+1 ring to avoid reusing one still held by present engine.
    acquire_semaphores_.resize(image_count + 1);
    for (auto& s : acquire_semaphores_)
        vkCreateSemaphore(ctx_->device(), &sem_info, nullptr, &s);
    acquire_sem_index_ = 0;

    // Render-finished semaphores: one per swapchain image.
    // Indexed by imageIndex from vkAcquireNextImageKHR.
    // Safe to reuse because acquire won't return that imageIndex until
    // the present engine is done with it (and therefore done with this semaphore).
    render_finished_semaphores_.resize(image_count);
    for (auto& s : render_finished_semaphores_)
        vkCreateSemaphore(ctx_->device(), &sem_info, nullptr, &s);

    // Fences: per frame-in-flight
    in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);
    for (auto& f : in_flight_fences_)
        vkCreateFence(ctx_->device(), &fence_info, nullptr, &f);
}

void Renderer::destroy_sync_objects() {
    for (auto s : acquire_semaphores_)
        vkDestroySemaphore(ctx_->device(), s, nullptr);
    acquire_semaphores_.clear();

    for (auto s : render_finished_semaphores_)
        vkDestroySemaphore(ctx_->device(), s, nullptr);
    render_finished_semaphores_.clear();

    for (auto f : in_flight_fences_)
        vkDestroyFence(ctx_->device(), f, nullptr);
    in_flight_fences_.clear();
}

void Renderer::on_swapchain_recreated() {
    vkDeviceWaitIdle(ctx_->device());
    destroy_sync_objects();
    create_sync_objects();
}

void Renderer::record_and_submit(PushConstants& pc, Graph& graph,
                                 bool& graph_changed, bool& recompile_requested) {
    // Wait for this frame slot's previous GPU work to finish.
    vkWaitForFences(ctx_->device(), 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);

    // Pick the next acquire semaphore from our N+1 ring.
    VkSemaphore acquire_sem = acquire_semaphores_[acquire_sem_index_];
    acquire_sem_index_ = (acquire_sem_index_ + 1) % (uint32_t)acquire_semaphores_.size();

    uint32_t image_index;
    VkResult result = vkAcquireNextImageKHR(ctx_->device(), swapchain_->handle(), UINT64_MAX,
                                            acquire_sem, VK_NULL_HANDLE, &image_index);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain_->recreate(swapchain_->extent().width, swapchain_->extent().height);
        on_swapchain_recreated();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    vkResetFences(ctx_->device(), 1, &in_flight_fences_[current_frame_]);

    VkCommandBuffer cmd = command_buffers_[current_frame_];
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &begin_info);

    imgui_->begin_frame();
    imgui_->render_ui(pc, graph, engine_->node_library(), graph_changed, recompile_requested);

    if (engine_->has_pipeline()) {
        // Transition engine output: UNDEFINED -> GENERAL
        VkImageMemoryBarrier2 to_general{};
        to_general.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_general.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        to_general.srcAccessMask = 0;
        to_general.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        to_general.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        to_general.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_general.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_general.image = engine_->output().image();
        to_general.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &to_general;
        vkCmdPipelineBarrier2(cmd, &dep);

        engine_->record_dispatch(cmd, pc);

        // Transition for blit: engine GENERAL->TRANSFER_SRC, swapchain UNDEFINED->TRANSFER_DST
        VkImageMemoryBarrier2 blit_barriers[2]{};
        blit_barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        blit_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        blit_barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        blit_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        blit_barriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        blit_barriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        blit_barriers[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blit_barriers[0].image = engine_->output().image();
        blit_barriers[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        blit_barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        blit_barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        blit_barriers[1].srcAccessMask = 0;
        blit_barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        blit_barriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        blit_barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        blit_barriers[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blit_barriers[1].image = swapchain_->images()[image_index];
        blit_barriers[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        dep.imageMemoryBarrierCount = 2;
        dep.pImageMemoryBarriers = blit_barriers;
        vkCmdPipelineBarrier2(cmd, &dep);

        // Blit compute output to swapchain
        VkImageBlit2 region{};
        region.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffsets[1] = {(int32_t)engine_->output().extent().width,
                                (int32_t)engine_->output().extent().height, 1};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.dstOffsets[1] = {(int32_t)swapchain_->extent().width,
                                (int32_t)swapchain_->extent().height, 1};

        VkBlitImageInfo2 blit{};
        blit.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
        blit.srcImage = engine_->output().image();
        blit.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blit.dstImage = swapchain_->images()[image_index];
        blit.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blit.regionCount = 1;
        blit.pRegions = &region;
        blit.filter = VK_FILTER_LINEAR;
        vkCmdBlitImage2(cmd, &blit);

        // Transition swapchain: TRANSFER_DST -> COLOR_ATTACHMENT_OPTIMAL (for ImGui)
        VkImageMemoryBarrier2 to_attach{};
        to_attach.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_attach.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
        to_attach.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        to_attach.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        to_attach.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        to_attach.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_attach.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_attach.image = swapchain_->images()[image_index];
        to_attach.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &to_attach;
        vkCmdPipelineBarrier2(cmd, &dep);
    } else {
        // No pipeline — transition swapchain to COLOR_ATTACHMENT for ImGui on black
        VkImageMemoryBarrier2 to_attach{};
        to_attach.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        to_attach.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        to_attach.srcAccessMask = 0;
        to_attach.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        to_attach.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        to_attach.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_attach.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        to_attach.image = swapchain_->images()[image_index];
        to_attach.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{};
        dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers = &to_attach;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // ImGui overlay
    imgui_->end_frame(cmd, swapchain_->image_views()[image_index],
                      swapchain_->extent().width, swapchain_->extent().height);

    // Transition swapchain: COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC_KHR
    VkImageMemoryBarrier2 to_present{};
    to_present.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    to_present.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    to_present.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    to_present.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
    to_present.dstAccessMask = 0;
    to_present.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    to_present.image = swapchain_->images()[image_index];
    to_present.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkDependencyInfo present_dep{};
    present_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    present_dep.imageMemoryBarrierCount = 1;
    present_dep.pImageMemoryBarriers = &to_present;
    vkCmdPipelineBarrier2(cmd, &present_dep);

    vkEndCommandBuffer(cmd);

    // The render_finished semaphore is indexed by imageIndex (per-swapchain-image).
    // This is the key fix: vkAcquireNextImageKHR won't return this imageIndex again
    // until the present engine is done with it, guaranteeing the semaphore is free.
    VkSemaphore render_sem = render_finished_semaphores_[image_index];

    // Submit — wait on acquire_sem, signal render_sem
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &acquire_sem;
    submit.pWaitDstStageMask = &wait_stage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &render_sem;

    vkQueueSubmit(ctx_->graphics_queue(), 1, &submit, in_flight_fences_[current_frame_]);

    // Present — wait on the same render_sem
    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &render_sem;
    present.swapchainCount = 1;
    VkSwapchainKHR sc = swapchain_->handle();
    present.pSwapchains = &sc;
    present.pImageIndices = &image_index;

    result = vkQueuePresentKHR(ctx_->graphics_queue(), &present);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        swapchain_->recreate(swapchain_->extent().width, swapchain_->extent().height);
        on_swapchain_recreated();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to present");
    }

    current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

} // namespace te
