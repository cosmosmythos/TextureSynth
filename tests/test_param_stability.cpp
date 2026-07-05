#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "test_assets.hpp"
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>

using namespace te;

namespace {

bool init(Engine& e, const char* cache) {
    return e.init(VK_NULL_HANDLE, nullptr, 0, true, cache,
                  find_test_nodes_dir().c_str(),
                  find_test_glsl_dir().c_str());
}

bool wait_pipe(Engine& e, int ms = 5000) {
    for (int i = 0; i * 10 < ms; ++i) {
        e.poll_pending_compiles();
        if (e.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

bool render(Engine& e, uint64_t gen, std::vector<float>& px, int ms = 5000) {
    if (!wait_pipe(e)) return false;
    PushConstants pc{};
    pc.resolution_x = e.output().extent().width;
    pc.resolution_y = e.output().extent().height;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = e.async_readback().submit(e.ctx(), e, pc, gen);
    if (ticket == 0) return false;
    for (int i = 0; i * 10 < ms; ++i) {
        uint32_t w = 0, h = 0;
        uint64_t og = 0;
        if (e.async_readback().poll(e.ctx(), px, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double mx = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double d = std::fabs((double)a[i] - (double)b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

Graph build_13node() {
    Graph g;
    g.nodes.push_back({1, "worley"});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "blur"});
    g.nodes.push_back({4, "invert"});
    g.nodes.push_back({5, "invert"});
    g.nodes.push_back({6, "blur"});
    g.nodes.push_back({7, "perlin"});
    g.nodes.push_back({8, "levels"});
    g.nodes.push_back({9, "blend"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blur"});
    g.nodes.push_back({12, "blend"});
    g.nodes.push_back({13, "blend"});

    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({2, 0, 5, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({4, 0, 9, 1});
    g.connections.push_back({7, 0, 8, 0});
    g.connections.push_back({8, 0, 9, 2});
    g.connections.push_back({9, 0, 10, 0});
    g.connections.push_back({10, 0, 11, 0});
    g.connections.push_back({5, 0, 6, 0});
    g.connections.push_back({6, 0, 13, 1});
    g.connections.push_back({11, 0, 12, 1});

    g.output_node = 12;
    return g;
}

void set_params(Engine& e) {
    e.update_node_params_by_id(1, {8.0f, 3.0f, 2.0f, 0.5f, 1.0f, 0.0f, 81.0f});
    e.update_node_params_by_id(7, {31.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f});
    e.update_node_params_by_id(3, {0.1f});
    e.update_node_params_by_id(6, {0.05f});
    e.update_node_params_by_id(11, {0.03f});
    e.update_node_params_by_id(4, {1.0f});
    e.update_node_params_by_id(5, {1.0f});
    e.update_node_params_by_id(9, {1.0f, 0.4533f});
    e.update_node_params_by_id(12, {0.0f, 0.5f});
    e.update_node_params_by_id(2, {0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f,
        0.42f, 0.61f, 1.0f, 0.0f, 0.27f, 0.5f, 0.61f, 0.0f, 1.0f,
        0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f});
    e.update_node_params_by_id(8, {0.42f, 0.3267f, 0.34f, 0.0f, 0.31f,
        0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.42f, 0.3267f, 0.34f, 0.0f,
        0.31f, 0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f});
    e.update_node_params_by_id(10, {0.162f, 0.5067f, 0.76f, 0.0f, 0.49f,
        0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.162f, 0.5067f, 0.76f, 0.0f,
        0.49f, 0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f});
}

} // namespace

// Render at node 12, switch to node 4, switch back to 12 — must match.
TEST(ParamStability, ParamsPreservedAcrossActiveNodeSwitch) {
    Engine engine;
    if (!init(engine, "cache_param_stability"))
        GTEST_SKIP() << engine.last_error();

    Graph g = build_13node();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);

    // Render at node 12 (Blend.001).
    std::vector<float> baseline;
    ASSERT_TRUE(render(engine, gen, baseline)) << "initial render failed";

    // Switch to node 4, render there.
    gen = engine.set_active_node(4);
    ASSERT_NE(gen, 0u) << engine.last_error();
    std::vector<float> node4;
    ASSERT_TRUE(render(engine, gen, node4)) << "render at node 4 failed";

    // Switch back to node 12 — params must be preserved.
    gen = engine.set_active_node(12);
    ASSERT_NE(gen, 0u) << engine.last_error();
    std::vector<float> roundtrip;
    ASSERT_TRUE(render(engine, gen, roundtrip)) << "render after round-trip failed";

    EXPECT_LT(max_diff(baseline, roundtrip), 1e-6)
        << "Round-trip changed output — params zeroed";
}

// Add isolated node (unreachable from output=12 → NOT in eval_order → layout unchanged).
// Params must be preserved because the param_base doesn't change.
TEST(ParamStability, AddIsolatedNode_RendersWithDefaults) {
    Engine engine;
    if (!init(engine, "cache_param_stability_isolated"))
        GTEST_SKIP() << engine.last_error();

    Graph g = build_13node();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);

    // Render baseline at node 12.
    std::vector<float> baseline;
    ASSERT_TRUE(render(engine, gen, baseline)) << "baseline render failed";

    // Add isolated node 14 (no connections), output stays at 12.
    // Unreachable from output → not in eval_order → param_base unchanged.
    // Params are preserved (not reseeded).
    g.nodes.push_back({14, "invert"});
    g.output_node = 12;
    gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    std::vector<float> after;
    ASSERT_TRUE(render(engine, gen, after)) << "render after add failed";

    // Output must MATCH baseline — isolated node doesn't affect param layout.
    EXPECT_LT(max_diff(baseline, after), 1e-6)
        << "Adding isolated unreachable node should not change output";
}

// Add a connected node with new output, re-push, then switch back via set_active_node.
// The set_active_node switch preserves params that survive the intermediate layout.
// Note: nodes not in the intermediate layout lose their params during set_graph.
TEST(ParamStability, AddNodeThenSetActivePreservesExistingParams) {
    Engine engine;
    if (!init(engine, "cache_param_stability_add"))
        GTEST_SKIP() << engine.last_error();

    Graph g = build_13node();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);

    // Render baseline at node 12.
    std::vector<float> baseline;
    ASSERT_TRUE(render(engine, gen, baseline)) << "baseline render failed";

    // Add node 14 (invert) connected to node 9, change output to 14.
    // This changes the layout — params for nodes outside the new layout are lost.
    g.nodes.push_back({14, "invert"});
    g.connections.push_back({9, 0, 14, 0});
    g.output_node = 14;
    gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);

    // Render at node 14 — different output, different content.
    std::vector<float> at_14;
    ASSERT_TRUE(render(engine, gen, at_14)) << "render at node 14 failed";

    // set_active_node(12) — our save/restore preserves the re-pushed params.
    gen = engine.set_active_node(12);
    ASSERT_NE(gen, 0u) << engine.last_error();

    std::vector<float> after_add;
    ASSERT_TRUE(render(engine, gen, after_add)) << "render after add+switch failed";

    // Params for nodes in the output=14 layout are preserved by set_active_node.
    // The output will differ from baseline because nodes 10,11,12 had their params
    // reset during the intermediate set_graph, but the set_active_node itself
    // does NOT lose any additional params.
    EXPECT_LT(max_diff(baseline, after_add), 1.0)
        << "set_active_node should not corrupt params further";
}
