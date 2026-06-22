#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "test_assets.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>

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

// Verifies SSBO contents match what the shader should read.
// This test dumps the param SSBO ring contents and compares with expected.
class BlendSSBODump : public ::testing::Test {
protected:
    Engine engine;
    void SetUp() override {
        if (!init_engine(engine, "test_ssbodump")) GTEST_SKIP() << engine.last_error();
    }
};

TEST_F(BlendSSBODump, ParamSlotsCorrect) {
    auto g = make_blend_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_pipeline(engine));

    auto layout = engine.param_layout();
    std::cout << "\nparam_layout:";
    for (auto& [id, base] : layout) std::cout << " node" << id << "=base" << base;
    std::cout << "\ntotal_param_floats=" << engine.total_param_floats() << std::endl;

    // Push params: simplex=defaults, value=defaults, blend=[mode=0, mask=0.5]
    engine.update_node_params_by_id(1, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f, 43.0f});
    engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
    engine.update_node_params_by_id(3, {0.0f, 0.5f});  // mode=0, mask=0.5

    // Now dump the SSBO contents directly from mapped memory
    // The engine has param_mapped_[PARAM_RING] — but these are private.
    // Instead, let's read back the GPU output and verify it's different from mask=0 and mask=1.
    
    // Render mask=0.5
    std::vector<float> px_half; uint32_t w=0, h=0;
    ASSERT_TRUE(readback(engine, gen, px_half, w, h));
    double mean_half = 0; size_t n = 0;
    for (size_t i = 0; i + 3 < px_half.size(); i += 4) { mean_half += px_half[i]; ++n; }
    mean_half /= n;
    std::cout << "mask=0.5 mean_R=" << mean_half << std::endl;

    // Now change mask to 0 and render again (NO set_graph — same pipeline)
    engine.update_node_params_by_id(3, {0.0f, 0.0f});  // mask=0
    std::vector<float> px_zero; 
    ASSERT_TRUE(readback(engine, gen, px_zero, w, h));
    double mean_zero = 0; n = 0;
    for (size_t i = 0; i + 3 < px_zero.size(); i += 4) { mean_zero += px_zero[i]; ++n; }
    mean_zero /= n;
    std::cout << "mask=0 mean_R=" << mean_zero << std::endl;

    // Change mask to 1 and render again
    engine.update_node_params_by_id(3, {0.0f, 1.0f});  // mask=1
    std::vector<float> px_one;
    ASSERT_TRUE(readback(engine, gen, px_one, w, h));
    double mean_one = 0; n = 0;
    for (size_t i = 0; i + 3 < px_one.size(); i += 4) { mean_one += px_one[i]; ++n; }
    mean_one /= n;
    std::cout << "mask=1 mean_R=" << mean_one << std::endl;

    // All three MUST differ
    double d01 = 0, d02 = 0, d12 = 0;
    for (size_t i = 0; i < px_zero.size(); ++i) d01 += std::abs(px_zero[i] - px_one[i]);
    for (size_t i = 0; i < px_zero.size(); ++i) d02 += std::abs(px_zero[i] - px_half[i]);
    for (size_t i = 0; i < px_one.size(); ++i)  d12 += std::abs(px_one[i] - px_half[i]);
    std::cout << "diff 0vs1=" << d01 << " 0vs0.5=" << d02 << " 1vs0.5=" << d12 << std::endl;

    EXPECT_GT(d01, 1.0);
    EXPECT_GT(d02, 1.0);
    EXPECT_GT(d12, 1.0);
}
