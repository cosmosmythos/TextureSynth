#include "viewer/Window.hpp"
#include "viewer/Swapchain.hpp"
#include "viewer/Renderer.hpp"
#include "viewer/ImGuiLayer.hpp"
#include "viewer/RenderDocCapture.hpp"
#include "engine/Engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace te;

int main() {
    try {
        std::cerr << "[viewer] init: rdoc\n";
        // RenderDoc capture API: must be initialized BEFORE the Vulkan
        // instance is created, so that when qrenderdoc attaches to this
        // process it can hook the instance creation. If RenderDoc is
        // not attached, init() is a no-op.
        RenderDocCapture rdoc;
        rdoc.init();

        std::cerr << "[viewer] init: window\n";
        Window window;
        if (!window.init(1280, 720, "Texture Engine Viewer")) {
            return -1;
        }

        uint32_t glfw_ext_count = 0;
        const char** glfw_exts = glfwGetRequiredInstanceExtensions(&glfw_ext_count);

        std::cerr << "[viewer] init: engine\n";
        Engine engine;
        // Asset dirs are relative to the working directory the viewer is launched from.
        // Launch from the repo root (the standard way: .\build\Release\viewer.exe) so
        // these resolve correctly.
        if (!engine.init(VK_NULL_HANDLE, glfw_exts, glfw_ext_count, false, "",
                         "shader_assets/nodes", "shader_assets/glsl")) {
            std::cerr << "FATAL: engine.init failed — check stderr for details\n";
            return -1;
        }

        VkSurfaceKHR surface = window.create_surface(engine.ctx().instance());

        std::cerr << "[viewer] init: swapchain\n";
        Swapchain swapchain;
        if (!swapchain.init(engine.ctx(), surface, window.width(), window.height())) {
            return -1;
        }

        std::cerr << "[viewer] init: imgui\n";
        ImGuiLayer imgui;
        if (!imgui.init(window, engine.ctx(), swapchain.image_format())) {
            return -1;
        }

        // Hand the capture wrapper to the renderer so it can start/end
        // captures around the command buffer recording when the user
        // clicks "Capture Frame" in the UI.
        std::cerr << "[viewer] init: renderer\n";
        Renderer renderer;
        if (!renderer.init(engine.ctx(), swapchain, engine, imgui, &rdoc)) {
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

        // Boot: set_graph is async (dispatches SPIR-V compile futures).
        // Submit the initial graph ONCE, then spin poll_pending_compiles until
        // has_pipeline() flips true. Don't keep re-submitting — each call bumps
        // the generation and supersedes the in-flight compile, so the engine
        // would never finish compiling.
        bool booting_ = true;
        bool initial_compile_submitted_ = false;
        int frame_count_ = 0;
        std::cerr << "[viewer] init: enter loop\n";

        while (!window.should_close()) {
            window.poll_events();
            if (frame_count_ < 3) {
                std::cerr << "[viewer] frame " << frame_count_ << "\n";
            }
            frame_count_++;

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

            if (booting_) {
                if (!initial_compile_submitted_) {
                    engine.set_graph(graph);
                    initial_compile_submitted_ = true;
                }
                if (engine.has_pipeline()) {
                    booting_ = false;
                }
            } else if (graph_changed || recompile_requested) {
                engine.set_graph(graph);
            }

            engine.tick_retired();

            // Surface the per-frame dispatch count to the UI so the user
            // can see at a glance whether the engine is actually running
            // or sitting idle (dirty_set empty -> 0 dispatches).
            imgui.set_dispatch_count(engine.last_dispatch_count());
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
