#include "viewer/Window.hpp"
#include "viewer/Swapchain.hpp"
#include "viewer/Renderer.hpp"
#include "viewer/ImGuiLayer.hpp"
#include "engine/Engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace te;

int main() {
    try {
        Window window;
        if (!window.init(1280, 720, "Texture Engine Viewer")) {
            return -1;
        }

        uint32_t glfw_ext_count = 0;
        const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

        Engine engine;
        if (!engine.init(VK_NULL_HANDLE, glfw_exts, glfw_ext_count, false, "", "", "")) {
            return -1;
        }

        VkSurfaceKHR surface = window.create_surface(engine.ctx().instance());

        Swapchain swapchain;
        if (!swapchain.init(engine.ctx(), surface, window.width(), window.height())) {
            return -1;
        }

        ImGuiLayer imgui;
        if (!imgui.init(window, engine.ctx(), swapchain.image_format())) {
            return -1;
        }

        Renderer renderer;
        if (!renderer.init(engine.ctx(), swapchain, engine, imgui)) {
            return -1;
        }

        PushConstants pc{};
        pc.resolution_x = engine.output().extent().width;
        pc.resolution_y = engine.output().extent().height;
        pc.seed = 42;
        pc.time = 0.0f;

        // Build default graph: Perlin[0] -> Invert[1]
        Graph graph;
        graph.nodes.push_back({0, "perlin"});
        graph.nodes.push_back({1, "invert"});
        graph.connections.push_back({0, 0, 1, 0}); // perlin output -> invert input
        graph.output_node = 1;                       // invert is the final output

        // Initial graph compile

        while (!window.should_close()) {
            window.poll_events();

            if (window.was_resized()) {
                window.wait_events_if_minimized();
                vkDeviceWaitIdle(engine.ctx().device());
                swapchain.recreate(window.width(), window.height());
                renderer.on_swapchain_recreated();
                window.reset_resized_flag();
            }

            auto current_time = glfwGetTime();
            pc.time = (float)current_time;

            engine.poll_pending_compiles();

            bool graph_changed = false;
            bool recompile_requested = false;
            renderer.record_and_submit(pc, graph, graph_changed, recompile_requested);

            if (graph_changed || recompile_requested) {
                engine.set_graph(graph);
            }

            engine.tick_retired();
        }

        vkDeviceWaitIdle(engine.ctx().device());
        renderer.shutdown();
        imgui.shutdown();
        swapchain.shutdown();
        vkDestroySurfaceKHR(engine.ctx().instance(), surface, nullptr);
        engine.shutdown();
        window.shutdown();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
