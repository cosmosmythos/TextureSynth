#pragma once
#include "engine/VulkanContext.hpp"
#include "engine/Engine.hpp"
#include "viewer/Swapchain.hpp"
#include "viewer/ImGuiLayer.hpp"
#include <vector>

namespace te {

class Renderer {
public:
    bool init(VulkanContext& ctx, Swapchain& swapchain, Engine& engine, ImGuiLayer& imgui);
    void shutdown();

    void on_swapchain_recreated();
    void record_and_submit(PushConstants& pc, Graph& graph,
                           bool& graph_changed, bool& recompile_requested);

private:
    void create_sync_objects();
    void destroy_sync_objects();
    void create_command_buffers();

    VulkanContext* ctx_ = nullptr;
    Swapchain* swapchain_ = nullptr;
    Engine* engine_ = nullptr;
    ImGuiLayer* imgui_ = nullptr;

    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;  // per frame-in-flight

    // Acquire semaphores: N+1 ring  (N = swapchain image count).
    // Cycled with acquire_sem_index_. Safe because at most N can be outstanding.
    std::vector<VkSemaphore> acquire_semaphores_;
    uint32_t acquire_sem_index_ = 0;

    // Render-finished semaphores: one per swapchain image, indexed by imageIndex.
    // The presentation engine holds these until it's done with the specific image,
    // and vkAcquireNextImageKHR won't return that imageIndex until it's released.
    std::vector<VkSemaphore> render_finished_semaphores_;

    // Fences: per frame-in-flight
    std::vector<VkFence> in_flight_fences_;

    uint32_t current_frame_ = 0;
};

} // namespace te
