#pragma once
#include "engine/VulkanContext.hpp"
#include "engine/PushConstants.hpp"
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include "viewer/Window.hpp"
#include <vulkan/vulkan.h>
#include <string>

namespace te {

// ImGuiLayer: frame-level UI for the dev viewer.
// Owns the imgui context and the per-frame windows:
//   - "Output": seed, resolution, Force Recompile, dispatch counter
//   - "Graph": a flat list of nodes. Each node shows:
//       * Mute / Bypass / Set as output / Remove
//       * Per-input socket row with Connect/Disconnect buttons
//       * Per-param row with SliderFloat (live, calls Engine::update_node_params_by_id)
//   - "Add Node": one button per NodeType, appended to the graph
//   - "RenderDoc": Capture Frame button (only when attached)
class ImGuiLayer {
public:
    bool init(Window& window, VulkanContext& ctx, VkFormat swapchain_format);
    void shutdown();

    void set_renderdoc_available(bool available) { rdoc_available_ = available; }
    void set_dispatch_count(uint64_t n) { last_dispatch_count_ = n; }

    void begin_frame();
    void render_ui(PushConstants& pc, Graph& graph, const NodeLibrary& lib,
                   bool& graph_changed, bool& recompile_requested,
                   bool& capture_requested);
    // clear_first=true: LOAD_OP_CLEAR with a black clear value (no engine output
    // was blitted). false: LOAD_OP_LOAD, preserve the blit result.
    void end_frame(VkCommandBuffer cmd, VkImageView target_view, uint32_t width, uint32_t height,
                   bool clear_first);

private:
    VulkanContext* ctx_ = nullptr;
    VkDescriptorPool imgui_pool_ = VK_NULL_HANDLE;
    bool rdoc_available_ = false;
    uint64_t last_dispatch_count_ = 0;
};

} // namespace te
