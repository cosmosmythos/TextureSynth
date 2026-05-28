#include "viewer/ImGuiLayer.hpp"
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <iostream>
#include <vector>
#include <algorithm>

namespace te {

// Helper to find a node instance by ID in a graph
static const NodeInstance* find_node_by_id(const Graph& g, uint32_t id) {
    for (auto& n : g.nodes)
        if (n.id == id) return &n;
    return nullptr;
}

bool ImGuiLayer::init(Window& window, VulkanContext& ctx, VkFormat swapchain_format) {
    ctx_ = &ctx;

    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
    };
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = std::size(pool_sizes);
    pool_info.pPoolSizes = pool_sizes;
    
    if (vkCreateDescriptorPool(ctx_->device(), &pool_info, nullptr, &imgui_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create ImGui descriptor pool\n";
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window.handle(), true);

    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.Instance = ctx_->instance();
    init_info.PhysicalDevice = ctx_->physical_device();
    init_info.Device = ctx_->device();
    init_info.QueueFamily = ctx_->graphics_family();
    init_info.Queue = ctx_->graphics_queue();
    init_info.DescriptorPool = imgui_pool_;
    init_info.MinImageCount = 2;
    init_info.ImageCount = 2;
    
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = {};
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain_format;

    ImGui_ImplVulkan_Init(&init_info);
    
    return true;
}

void ImGuiLayer::shutdown() {
    if (ctx_) {
        vkDeviceWaitIdle(ctx_->device());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        vkDestroyDescriptorPool(ctx_->device(), imgui_pool_, nullptr);
        ctx_ = nullptr;
    }
}

void ImGuiLayer::begin_frame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render_ui(PushConstants& pc, Graph& graph, const NodeLibrary& lib,
                           bool& graph_changed, bool& recompile_requested) {
    graph_changed = false;
    recompile_requested = false;

    ImGui::Begin("Node Graph");

    ImGui::Text("Resolution: %ux%u", pc.resolution_x, pc.resolution_y);
    int seed = (int)pc.seed;
    if (ImGui::InputInt("Seed", &seed)) {
        pc.seed = (uint32_t)std::max(0, seed);
    }

    ImGui::Separator();
    ImGui::Text("Nodes (%zu)", graph.nodes.size());
    ImGui::Separator();

    // Calculate parameter base slots (same logic as GraphCompiler)
    std::unordered_map<uint32_t, int> param_base;
    int slot = 0;
    // We need topo order, but for UI purposes just iterate in node order
    for (auto& node : graph.nodes) {
        auto* type = lib.find(node.type_id);
        if (!type) continue;
        param_base[node.id] = slot;
        slot += (int)type->params.size();
    }

    // Render per-node UI
    for (size_t ni = 0; ni < graph.nodes.size(); ni++) {
        auto& node = graph.nodes[ni];
        auto* type = lib.find(node.type_id);
        if (!type) continue;

        ImGui::PushID((int)node.id);

        bool is_output = (node.id == graph.output_node);
        std::string header = type->display_name + " [" + std::to_string(node.id) + "]";
        if (is_output) header += " (OUTPUT)";

        if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            // Inputs
            for (size_t i = 0; i < type->inputs.size(); i++) {
                bool connected = false;
                for (auto& c : graph.connections) {
                    if (c.dst_node == node.id && c.dst_socket == (uint32_t)i) {
                        auto* src = find_node_by_id(graph, c.src_node);
                        if (src) {
                            auto* src_type = lib.find(src->type_id);
                            ImGui::TextColored({0.5f, 0.8f, 0.5f, 1.0f}, "  <- %s from %s[%u]",
                                type->inputs[i].name.c_str(),
                                src_type ? src_type->display_name.c_str() : "?",
                                c.src_node);
                        }
                        connected = true;
                        break;
                    }
                }
                if (!connected) {
                    ImGui::TextColored({0.6f, 0.6f, 0.6f, 1.0f}, "  <- %s (unconnected)",
                        type->inputs[i].name.c_str());
                }
            }

            // Parameters (sliders mapped to hot_params)
            int base = param_base[node.id];
            for (size_t pi = 0; pi < type->params.size(); pi++) {
                auto& p = type->params[pi];
                ImGui::SliderFloat(p.name.c_str(), &pc.hot_params[base + pi], p.min_value, p.max_value);
            }

            // Set as output button
            if (!is_output) {
                if (ImGui::SmallButton("Set as Output")) {
                    graph.output_node = node.id;
                    graph_changed = true;
                }
                ImGui::SameLine();
            }

            // Remove button (only if not the last node)
            if (graph.nodes.size() > 1) {
                if (ImGui::SmallButton("Remove")) {
                    // Remove connections involving this node
                    graph.connections.erase(
                        std::remove_if(graph.connections.begin(), graph.connections.end(),
                            [&](const Connection& c) {
                                return c.src_node == node.id || c.dst_node == node.id;
                            }),
                        graph.connections.end());
                    // If output, reassign
                    if (is_output && !graph.nodes.empty()) {
                        graph.output_node = graph.nodes[0].id;
                    }
                    graph.nodes.erase(graph.nodes.begin() + ni);
                    graph_changed = true;
                    ImGui::PopID();
                    break; // iterator invalidated
                }
            }
        }
        ImGui::PopID();
    }

    ImGui::Separator();
    ImGui::Text("Add Node:");

    // Add node buttons for each type in library
    for (auto& [type_id, type] : lib.all()) {
        if (ImGui::Button(type.display_name.c_str())) {
            uint32_t new_id = 0;
            for (auto& n : graph.nodes)
                new_id = std::max(new_id, n.id);
            new_id++;

            graph.nodes.push_back({new_id, type_id});

            // Auto-connect: if this node has inputs and there's a current output node,
            // insert between the last node and output
            if (!type.inputs.empty() && graph.nodes.size() > 1) {
                // Connect previous output node's output[0] to new node's input[0]
                graph.connections.push_back({graph.output_node, 0, new_id, 0});
                graph.output_node = new_id;
            } else if (type.inputs.empty()) {
                // Generator node — if first node, set as output
                if (graph.nodes.size() == 1) {
                    graph.output_node = new_id;
                }
            }
            graph_changed = true;
        }
        ImGui::SameLine();
    }
    ImGui::NewLine();

    ImGui::Separator();
    if (ImGui::Button("Force Recompile")) {
        recompile_requested = true;
    }

    ImGui::End();
}

void ImGuiLayer::end_frame(VkCommandBuffer cmd, VkImageView target_view, uint32_t width, uint32_t height) {
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = target_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; 
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    render_info.renderArea.offset = {0, 0};
    render_info.renderArea.extent = {width, height};
    render_info.layerCount = 1;
    render_info.colorAttachmentCount = 1;
    render_info.pColorAttachments = &color_attachment;

    vkCmdBeginRendering(cmd, &render_info);
    ImGui_ImplVulkan_RenderDrawData(draw_data, cmd);
    vkCmdEndRendering(cmd);
}

} // namespace te
