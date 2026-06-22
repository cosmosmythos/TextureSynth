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
NodeLibrary load_lib() {
    NodeLibrary lib;
    std::string err;
    NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    return lib;
}
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
}

class BlendCrossChain : public ::testing::Test {
protected:
    Engine engine;
    void SetUp() override {
        if (!init_engine(engine, "test_blend_cross")) GTEST_SKIP() << engine.last_error();
    }
};

// Simplex(1) -> blend.a, Value(2) -> blend.b, output=blend(3)
// Push params, mask=0 -> expect A (simplex ~0.5)
// Push params, mask=1 -> expect B (value ~0.5)
// Push params, mask=0.5 -> expect mix ~0.5
TEST_F(BlendCrossChain, MaskControlsOutput) {
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "value"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});
    g.connections.push_back({2, 0, 3, 2});
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_pipeline(engine));

    // Push noise params
    engine.update_node_params_by_id(1, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f, 43.0f});
    engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});

    // Test mask=0 -> should be A (simplex)
    engine.update_node_params_by_id(3, {0.0f, 0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px0; uint32_t w=0, h=0;
    ASSERT_TRUE(readback(engine, gen, px0, w, h));
    double m0 = mean_r(px0);
    std::cout << "mask=0 mean_R=" << m0 << std::endl;

    // Test mask=1 -> should be B (value)
    engine.update_node_params_by_id(3, {0.0f, 1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px1;
    ASSERT_TRUE(readback(engine, gen, px1, w, h));
    double m1 = mean_r(px1);
    std::cout << "mask=1 mean_R=" << m1 << std::endl;

    // Test mask=0.5 -> mix
    engine.update_node_params_by_id(3, {0.0f, 0.5f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px2;
    ASSERT_TRUE(readback(engine, gen, px2, w, h));
    double m2 = mean_r(px2);
    std::cout << "mask=0.5 mean_R=" << m2 << std::endl;

    // The three outputs MUST differ
    double d01 = 0, d02 = 0, d12 = 0;
    for (size_t i = 0; i < px0.size() && i < px1.size(); ++i)
        d01 += std::abs(px0[i] - px1[i]);
    for (size_t i = 0; i < px0.size() && i < px2.size(); ++i)
        d02 += std::abs(px0[i] - px2[i]);
    for (size_t i = 0; i < px1.size() && i < px2.size(); ++i)
        d12 += std::abs(px1[i] - px2[i]);
    std::cout << "diff mask0vs1=" << d01 << " mask0vs0.5=" << d02 << " mask1vs0.5=" << d12 << std::endl;

    EXPECT_GT(d01, 1.0) << "mask=0 and mask=1 produced identical output";
    EXPECT_GT(d02, 1.0) << "mask=0 and mask=0.5 produced identical output";
    EXPECT_GT(d12, 1.0) << "mask=1 and mask=0.5 produced identical output";
}
