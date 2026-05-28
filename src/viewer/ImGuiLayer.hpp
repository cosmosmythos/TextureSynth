#pragma once
#include "engine/VulkanContext.hpp"
#include "engine/PushConstants.hpp"
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include "viewer/Window.hpp"
#include <vulkan/vulkan.h>

namespace te {

class ImGuiLayer {
public:
    bool init(Window& window, VulkanContext& ctx, VkFormat swapchain_format);
    void shutdown();

    void begin_frame();
    void render_ui(PushConstants& pc, Graph& graph, const NodeLibrary& lib,
                   bool& graph_changed, bool& recompile_requested);
    void end_frame(VkCommandBuffer cmd, VkImageView target_view, uint32_t width, uint32_t height);

private:
    VulkanContext* ctx_ = nullptr;
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
};

} // namespace te
