// Regression: Blend.001 with both A and B connected.
// When both inputs are connected, mask modulation breaks — output is constant.
// This test replicates the exact graph from the Blender scene and verifies
// that mask=0 outputs B, mask=1 outputs A, mask=0.5 outputs a mix.
#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "test_assets.hpp"
#include <thread>
#include <chrono>
#include <cmath>

using namespace te;

namespace {

NodeLibrary load_real_lib() {
    NodeLibrary lib;
    std::string err;
    int n = NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    EXPECT_GT(n, 0) << "failed to load real nodes: " << err;
    return lib;
}

bool init_engine(Engine& engine) {
    return engine.init(VK_NULL_HANDLE, nullptr, 0, true, "cache_blend_dual",
                       find_test_nodes_dir().c_str(),
                       find_test_glsl_dir().c_str());
}

bool wait_for_pipeline(Engine& engine, int timeout_ms = 5000) {
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        engine.poll_pending_compiles();
        if (engine.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return engine.has_pipeline();
}

bool render(Engine& engine, uint64_t gen,
            std::vector<float>& px, uint32_t& w, uint32_t& h,
            int timeout_ms = 5000) {
    if (!wait_for_pipeline(engine)) return false;
    PushConstants pc{};
    pc.resolution_x = 64; pc.resolution_y = 64;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    if (ticket == 0) return false;
    uint64_t og = 0;
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        if (engine.async_readback().poll(engine.ctx(), px, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

float mean_r(const std::vector<float>& px) {
    double sum = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) sum += px[i];
    return (float)(sum / (px.size() / 4));
}

// Build the exact graph from the Blender scene.
// Node IDs map to stable_id() values, but any unique IDs work.
//
// Graph topology:
//   Worley(1) -> Levels(2) -> Blur(3) -> Invert(4) -> Blend(9).A
//   Levels(2) -> Invert.001(5) -> Blur.001(6) -> Blend.001(12).B
//   Perlin(7) -> Levels.001(8) -> Blend(9).B
//   Blend(9) -> Levels.002(10) -> Blur.002(11) -> Blend.001(12).A
//   Output = Blend.001(12)
Graph build_full_graph() {
    Graph g;
    g.nodes.push_back({1,  "worley"});
    g.nodes.push_back({2,  "levels"});
    g.nodes.push_back({3,  "blur"});
    g.nodes.push_back({4,  "invert"});
    g.nodes.push_back({5,  "invert"});
    g.nodes.push_back({6,  "blur"});
    g.nodes.push_back({7,  "perlin"});
    g.nodes.push_back({8,  "levels"});
    g.nodes.push_back({9,  "blend"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blur"});
    g.nodes.push_back({12, "blend"});

    // Connections (from Blender introspection):
    g.connections.push_back({1, 0, 2, 0});   // Worley -> Levels.in[0]
    g.connections.push_back({2, 0, 3, 0});   // Levels -> Blur.in[0]
    g.connections.push_back({2, 0, 5, 1});   // Levels -> Invert.001.in[1] (Color)
    g.connections.push_back({3, 0, 4, 1});   // Blur -> Invert.in[1] (Color)
    g.connections.push_back({4, 0, 9, 1});   // Invert -> Blend.in[1] (A)
    g.connections.push_back({7, 0, 8, 0});   // Perlin -> Levels.001.in[0]
    g.connections.push_back({8, 0, 9, 2});   // Levels.001 -> Blend.in[2] (B)
    g.connections.push_back({9, 0, 10, 0});  // Blend -> Levels.002.in[0]
    g.connections.push_back({10, 0, 11, 0}); // Levels.002 -> Blur.002.in[0]
    g.connections.push_back({5, 0, 6, 0});   // Invert.001 -> Blur.001.in[0]
    g.connections.push_back({6, 0, 12, 2});  // Blur.001 -> Blend.001.in[2] (B)
    g.connections.push_back({11, 0, 12, 1}); // Blur.002 -> Blend.001.in[1] (A)

    g.output_node = 12;
    return g;
}

void set_all_params(Engine& engine) {
    // Exact params from Blender introspection:
    engine.update_node_params_by_id(1, {8.0f, 3.0f, 2.0f, 0.5f, 1.0f, 0.0f, 81.0f});
    engine.update_node_params_by_id(7, {31.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f});
    engine.update_node_params_by_id(3, {0.1f});
    engine.update_node_params_by_id(6, {0.05f});
    engine.update_node_params_by_id(11, {0.03f});
    engine.update_node_params_by_id(4, {1.0f});
    engine.update_node_params_by_id(5, {1.0f});
    engine.update_node_params_by_id(9, {1.0f, 0.4533f});
    engine.update_node_params_by_id(12, {0.0f, 0.5f});
    // Levels params (26 floats each, all R/G/B/A channel curves):
    std::vector<float> levels_p = {0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f,
        0.42f, 0.61f, 1.0f, 0.0f, 0.27f, 0.5f, 0.61f, 0.0f, 1.0f,
        0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(2, levels_p);
    std::vector<float> levels001_p = {0.42f, 0.3267f, 0.34f, 0.0f, 0.31f,
        0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.42f, 0.3267f, 0.34f, 0.0f,
        0.31f, 0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(8, levels001_p);
    std::vector<float> levels002_p = {0.162f, 0.5067f, 0.76f, 0.0f, 0.49f,
        0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.162f, 0.5067f, 0.76f, 0.0f,
        0.49f, 0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(10, levels002_p);
}

} // namespace


// Verify mask modulation — R should change between mask=0 and mask=1.
TEST(BlendDualInput, FullGraph_MaskModulation) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    Graph g = build_full_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    set_all_params(engine);

    // mask=0
    engine.update_node_params_by_id(12, {0.0f, 0.0f});
    std::vector<float> px0;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(render(engine, gen, px0, w, h));
    float r0 = mean_r(px0);

    // mask=1
    engine.update_node_params_by_id(12, {0.0f, 1.0f});
    std::vector<float> px1;
    ASSERT_TRUE(render(engine, gen, px1, w, h));
    float r1 = mean_r(px1);

    // mask=0.5
    engine.update_node_params_by_id(12, {0.0f, 0.5f});
    std::vector<float> px05;
    ASSERT_TRUE(render(engine, gen, px05, w, h));
    float r05 = mean_r(px05);

    std::cout << "Mask modulation: mask=0 -> R=" << r0
              << ", mask=0.5 -> R=" << r05
              << ", mask=1 -> R=" << r1 << std::endl;

    // R should differ between mask=0 and mask=1 (unless A==B which is unlikely).
    EXPECT_NE(r0, r1) << "mask=0 and mask=1 produced identical output — mask not modulating!";
    // mask=0.5 should be between mask=0 and mask=1 (for MIX mode).
    float lo = std::min(r0, r1);
    float hi = std::max(r0, r1);
    EXPECT_GE(r05, lo - 0.02f) << "mask=0.5 below both extremes";
    EXPECT_LE(r05, hi + 0.02f) << "mask=0.5 above both extremes";
}
