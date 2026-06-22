#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "test_assets.hpp"
#include <iostream>
#include <thread>
#include <chrono>

using namespace te;

namespace {
bool init_engine(Engine& e, const char* name) {
    return e.init(VK_NULL_HANDLE, nullptr, 0, true, name,
                  find_test_nodes_dir().c_str(), find_test_glsl_dir().c_str());
}
bool wait_pipeline(Engine& e, int ms = 5000) {
    for (int i = 0; i * 10 < ms; ++i) {
        e.poll_pending_compiles();
        if (e.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}
bool readback(Engine& e, uint64_t gen, std::vector<float>& px, uint32_t& w, uint32_t& h, int ms = 5000) {
    PushConstants pc{};
    pc.resolution_x = 128; pc.resolution_y = 128; pc.seed = 1; pc.time = 0;
    uint64_t ticket = e.async_readback().submit(e.ctx(), e, pc, gen);
    if (!ticket) return false;
    uint64_t og = 0;
    for (int i = 0; i * 10 < ms; ++i) {
        if (e.async_readback().poll(e.ctx(), px, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}
double mean_r(const std::vector<float>& px) {
    double s = 0; size_t n = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) { s += px[i]; ++n; }
    return n ? s / n : -1;
}
Graph make_blend_graph() {
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "value"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});
    g.connections.push_back({2, 0, 3, 2});
    g.output_node = 3;
    return g;
}
}

// Mimics the Python test: set_graph + push params + readback, repeated 3 times.
// The C++ test that passes does NOT call set_graph between reads.
// This test DOES call set_graph each time (like Python).
class BlendPythonRepro : public ::testing::Test {
protected:
    Engine engine;
    void SetUp() override {
        if (!init_engine(engine, "test_py_repro")) GTEST_SKIP() << engine.last_error();
    }
};

TEST_F(BlendPythonRepro, PerRenderSetGraph) {
    // Render 1: mask=0
    {
        auto g = make_blend_graph();
        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        ASSERT_TRUE(wait_pipeline(engine));
        engine.update_node_params_by_id(1, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f, 43.0f});
        engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
        engine.update_node_params_by_id(3, {0.0f, 0.0f});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        std::vector<float> px; uint32_t w = 0, h = 0;
        ASSERT_TRUE(readback(engine, gen, px, w, h));
        std::cout << "set_graph-per-call mask=0 mean_R=" << mean_r(px) << std::endl;
    }
    // Render 2: mask=1
    {
        auto g = make_blend_graph();
        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        ASSERT_TRUE(wait_pipeline(engine));
        engine.update_node_params_by_id(1, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f, 43.0f});
        engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
        engine.update_node_params_by_id(3, {0.0f, 1.0f});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        std::vector<float> px; uint32_t w = 0, h = 0;
        ASSERT_TRUE(readback(engine, gen, px, w, h));
        std::cout << "set_graph-per-call mask=1 mean_R=" << mean_r(px) << std::endl;
    }
}
