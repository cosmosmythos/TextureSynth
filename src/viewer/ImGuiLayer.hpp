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

    // renderdoc_available: enable the "Capture Frame" button. Set to
    // false to hide it entirely. Defaults to false.
    void set_renderdoc_available(bool available) { rdoc_available_ = available; }

    void begin_frame();
    void render_ui(PushConstants& pc, Graph& graph, const NodeLibrary& lib,
                   bool& graph_changed, bool& recompile_requested,
                   bool& capture_requested);
    // clear_first=true: LOAD_OP_CLEAR with a black clear value. Use when the
    // caller has NOT blitted a fullscreen image into the swapchain, otherwise
    // uninitialized garbage plus LOAD_OP_LOAD causes accumulated ghosts from
    // previous frames. clear_first=false: LOAD_OP_LOAD — preserve whatever
    // was written into the image (typically the engine's blit result).
    void end_frame(VkCommandBuffer cmd, VkImageView target_view, uint32_t width, uint32_t height,
                   bool clear_first);

private:
    VulkanContext* ctx_ = nullptr;
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
    bool rdoc_available_ = false;
};

} // namespace te
