#include "viewer/ImGuiLayer.hpp"
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <algorithm>
#include <set>
#include <cstdio>

namespace te {

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

namespace {

// Find the connection feeding (dst_node, dst_socket), if any.
const Connection* find_incoming(const Graph& g, NodeId dst, uint32_t socket) {
    for (const auto& c : g.connections) {
        if (c.dst_node == dst && c.dst_socket == socket) return &c;
    }
    return nullptr;
}

const NodeType* find_type(const NodeLibrary& lib, const std::string& type_id) {
    const auto& all = lib.all();
    auto it = all.find(type_id);
    return (it == all.end()) ? nullptr : &it->second;
}

const NodeInstance* find_node(const Graph& g, NodeId id) {
    for (const auto& n : g.nodes) {
        if (n.id == id) return &n;
    }
    return nullptr;
}

} // anonymous

void ImGuiLayer::render_ui(PushConstants& pc, Graph& graph, const NodeLibrary& lib,
                           bool& graph_changed, bool& recompile_requested,
                           bool& capture_requested) {
    graph_changed = false;
    recompile_requested = false;
    capture_requested = false;

    // ---- Output window: globals + dispatch counter + Force Recompile ----
    ImGui::Begin("Output");
    ImGui::Text("Resolution: %ux%u", pc.resolution_x, pc.resolution_y);
    int seed = static_cast<int>(pc.seed);
    if (ImGui::InputInt("Seed", &seed)) {
        pc.seed = static_cast<uint32_t>(std::max(0, seed));
    }
    if (last_dispatch_count_ == 0) {
        ImGui::TextDisabled("Last frame dispatches: 0 (static graph; engine is idle)");
    } else {
        ImGui::Text("Last frame dispatches: %llu", static_cast<unsigned long long>(last_dispatch_count_));
    }
    if (ImGui::Button("Force Recompile")) {
        recompile_requested = true;
    }
    ImGui::Separator();
    if (rdoc_available_) {
        ImGui::TextColored({0.4f, 0.8f, 1.0f, 1.0f}, "RenderDoc: attached");
        if (ImGui::Button("Capture Frame")) {
            capture_requested = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Triggers a single-frame capture in the attached RenderDoc.");
        }
    } else {
        ImGui::TextDisabled("RenderDoc: not attached");
    }
    ImGui::End();

    // ---- Add Node window: one button per NodeType ----
    ImGui::Begin("Add Node");
    for (const auto& [type_id, type] : lib.all()) {
        ImGui::PushID(type_id.c_str());
        if (ImGui::Button(type.display_name.c_str())) {
            uint32_t new_id = 0;
            for (const auto& n : graph.nodes) {
                if (static_cast<uint32_t>(n.id) > new_id) new_id = static_cast<uint32_t>(n.id);
            }
            new_id++;
            NodeInstance ni;
            ni.id = new_id;
            ni.type_id = type_id;
            graph.nodes.push_back(ni);
            if (graph.nodes.size() == 1) graph.output_node = new_id;
            graph_changed = true;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", type.description.c_str());
        }
        ImGui::PopID();
    }
    ImGui::End();

    // ---- Graph window: per-node list, with edit affordances ----
    ImGui::Begin("Graph");
    // Snapshot of nodes (we may erase while iterating).
    for (size_t i = 0; i < graph.nodes.size(); ++i) {
        NodeId nid = graph.nodes[i].id;
        const NodeType* type = find_type(lib, graph.nodes[i].type_id);

        ImGui::PushID(static_cast<int>(nid));

        std::string header = std::string(type ? type->display_name.c_str() : graph.nodes[i].type_id.c_str())
                           + " [#" + std::to_string(nid) + "]";
        if (graph.output_node == nid) header += "  (OUTPUT)";
        if (graph.nodes[i].muted)    header += "  (muted)";
        if (graph.nodes[i].bypassed) header += "  (bypassed)";

        if (ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            // Node-level actions
            if (ImGui::SmallButton("Set as output")) {
                graph.output_node = nid;
                graph_changed = true;
            }
            ImGui::SameLine();
            bool muted = graph.nodes[i].muted;
            if (ImGui::Checkbox("Mute", &muted)) {
                graph.nodes[i].muted = muted;
                graph_changed = true;
            }
            ImGui::SameLine();
            bool bypassed = graph.nodes[i].bypassed;
            if (ImGui::Checkbox("Bypass", &bypassed)) {
                graph.nodes[i].bypassed = bypassed;
                graph_changed = true;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Remove")) {
                graph.nodes.erase(graph.nodes.begin() + i);
                graph.connections.erase(
                    std::remove_if(graph.connections.begin(), graph.connections.end(),
                        [&](const Connection& c) {
                            return c.src_node == nid || c.dst_node == nid;
                        }),
                    graph.connections.end());
                if (graph.output_node == nid && !graph.nodes.empty()) {
                    graph.output_node = graph.nodes.front().id;
                }
                graph_changed = true;
                ImGui::PopID();
                break;
            }

            // Per-input socket row: Connect / Disconnect.
            if (type) {
                for (uint32_t s = 0; s < type->inputs.size(); ++s) {
                    ImGui::PushID(static_cast<int>(s));
                    const Connection* inc = find_incoming(graph, nid, s);
                    const char* sock_name = type->inputs[s].name.c_str();
                    if (inc) {
                        const NodeInstance* src_ni = find_node(graph, inc->src_node);
                        const NodeType* src_type = src_ni ? find_type(lib, src_ni->type_id) : nullptr;
                        std::string src_label = src_ni ? std::to_string(inc->src_node) : "?";
                        if (src_type) {
                            uint32_t ss = inc->src_socket;
                            src_label += " (" + std::string(src_type->display_name);
                            if (ss < src_type->outputs.size()) {
                                src_label += "." + src_type->outputs[ss].name;
                            }
                            src_label += ")";
                        }
                        ImGui::Text("%s <- %s", sock_name, src_label.c_str());
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Disconnect")) {
                            graph.connections.erase(
                                std::remove_if(graph.connections.begin(), graph.connections.end(),
                                    [&](const Connection& c) {
                                        return c.src_node == inc->src_node
                                            && c.dst_node == nid
                                            && c.dst_socket == s;
                                    }),
                                graph.connections.end());
                            graph_changed = true;
                        }
                    } else {
                        ImGui::Text("%s (unconnected)", sock_name);
                        ImGui::SameLine();
                        if (ImGui::SmallButton("Connect...")) {
                            ImGui::OpenPopup("connect_popup");
                        }
                        if (ImGui::BeginPopup("connect_popup")) {
                            for (size_t j = 0; j < graph.nodes.size(); ++j) {
                                if (graph.nodes[j].id == nid) continue;
                                const NodeType* src_type = find_type(lib, graph.nodes[j].type_id);
                                if (!src_type) continue;
                                for (uint32_t ss = 0; ss < src_type->outputs.size(); ++ss) {
                                    std::string label = std::to_string(graph.nodes[j].id) + "."
                                                      + src_type->display_name + "."
                                                      + src_type->outputs[ss].name;
                                    if (ImGui::MenuItem(label.c_str())) {
                                        Connection c;
                                        c.src_node = graph.nodes[j].id;
                                        c.src_socket = ss;
                                        c.dst_node = nid;
                                        c.dst_socket = s;
                                        graph.connections.push_back(c);
                                        graph_changed = true;
                                    }
                                }
                            }
                            ImGui::EndPopup();
                        }
                    }
                    ImGui::PopID();
                }
            }

            // Per-param display (read-only; live edit lives in the JSON graph
            // via a future editor or via update_node_params_by_id from a
            // separate ParamEditor window).
            if (type && !type->params.empty()) {
                ImGui::Separator();
                ImGui::TextDisabled("Parameters (edit via future ParamEditor):");
                for (const auto& pd : type->params) {
                    ImGui::BulletText("%s  [%.3f .. %.3f]  default %.3f",
                                      pd.display_name.c_str(),
                                      pd.min_value, pd.max_value, pd.default_value);
                }
            }
        }
        ImGui::PopID();
    }
    ImGui::End();
}

void ImGuiLayer::end_frame(VkCommandBuffer cmd, VkImageView target_view, uint32_t width, uint32_t height,
                           bool clear_first) {
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();

    VkClearValue clear_value{};
    clear_value.color = {{0.0f, 0.0f, 0.0f, 1.0f}};

    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = target_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = clear_first ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    if (clear_first) {
        color_attachment.clearValue = clear_value;
    }

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
