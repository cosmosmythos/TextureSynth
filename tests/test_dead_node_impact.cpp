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
#include <iostream>
#include <vector>

using namespace te;

namespace {

NodeLibrary load_lib() {
    NodeLibrary lib;
    std::string err;
    NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    return lib;
}

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

double mean_r(const std::vector<float>& px) {
    double s = 0; size_t n = px.size() / 4;
    for (size_t i = 0; i + 3 < px.size(); i += 4) s += px[i];
    return n ? s / n : 0;
}

// Graph WITHOUT dead Blend.002 (12 nodes).
// All nodes reachable from output=12.
Graph build_graph_without_dead() {
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
    g.connections.push_back({6, 0, 12, 2});
    g.connections.push_back({11, 0, 12, 1});

    g.output_node = 12;
    return g;
}

// Graph WITH dead Blend.002 (13 nodes).
// Adds node 13 (blend) and link 6->13. Node 13 is unreachable from output.
Graph build_graph_with_dead() {
    Graph g = build_graph_without_dead();
    g.nodes.push_back({13, "blend"});
    g.connections.push_back({6, 0, 13, 1});
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
    std::vector<float> lp = {0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f,
        0.42f, 0.61f, 1.0f, 0.0f, 0.27f, 0.5f, 0.61f, 0.0f, 1.0f,
        0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f};
    e.update_node_params_by_id(2, lp);
    std::vector<float> l1 = {0.42f, 0.3267f, 0.34f, 0.0f, 0.31f,
        0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.42f, 0.3267f, 0.34f, 0.0f,
        0.31f, 0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    e.update_node_params_by_id(8, l1);
    std::vector<float> l2 = {0.162f, 0.5067f, 0.76f, 0.0f, 0.49f,
        0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.162f, 0.5067f, 0.76f, 0.0f,
        0.49f, 0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    e.update_node_params_by_id(10, l2);
}

} // namespace


// Test: Does adding a dead unreachable node change the output of the live graph?
// Builds the exact Blend.001 graph twice: once without Blend.002, once with.
// Both should produce identical pixel output for Blend.001.
TEST(DeadNodeImpact, UnreachableNodeDoesNotChangeOutput) {
    Engine engine;
    if (!init(engine, "cache_dead_node_impact"))
        GTEST_SKIP() << engine.last_error();

    // --- Render WITHOUT dead node ---
    Graph g_clean = build_graph_without_dead();
    uint64_t gen_clean = engine.set_graph(g_clean);
    ASSERT_NE(gen_clean, 0u) << engine.last_error();
    set_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 0.5f});

    std::vector<float> px_clean;
    ASSERT_TRUE(render(engine, gen_clean, px_clean));
    double mean_clean = mean_r(px_clean);
    std::cout << "WITHOUT dead node: mean R = " << mean_clean << std::endl;

    // --- Render WITH dead node ---
    Graph g_dead = build_graph_with_dead();
    uint64_t gen_dead = engine.set_graph(g_dead);
    ASSERT_NE(gen_dead, 0u) << engine.last_error();
    set_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 0.5f});

    std::vector<float> px_dead;
    ASSERT_TRUE(render(engine, gen_dead, px_dead));
    double mean_dead = mean_r(px_dead);
    std::cout << "WITH dead node:    mean R = " << mean_dead << std::endl;

    // --- Compare ---
    double diff = max_diff(px_clean, px_dead);
    std::cout << "Max pixel diff:    " << diff << std::endl;

    // The dead node is unreachable — output MUST be identical.
    EXPECT_EQ(px_clean.size(), px_dead.size())
        << "pixel buffer size mismatch";
    EXPECT_NEAR(mean_clean, mean_dead, 1e-6)
        << "Dead node changed mean output!";
    EXPECT_LT(diff, 1e-6)
        << "Dead node changed ANY pixel! Max diff = " << diff;

    engine.shutdown();
}


// Test: Does adding a dead node change the output of a SWITCHED active node?
// Set active to node 4 (Invert) — same graph, with/without dead Blend.002.
TEST(DeadNodeImpact, UnreachableNodeDoesNotChangeSwitchedOutput) {
    Engine engine;
    if (!init(engine, "cache_dead_node_impact_switch"))
        GTEST_SKIP() << engine.last_error();

    // --- Without dead, switch to invert ---
    Graph g_clean = build_graph_without_dead();
    uint64_t gen_clean = engine.set_graph(g_clean);
    ASSERT_NE(gen_clean, 0u) << engine.last_error();
    set_params(engine);

    uint64_t gen_inv = engine.set_active_node(4);
    ASSERT_NE(gen_inv, 0u) << engine.last_error();
    std::vector<float> px_clean;
    ASSERT_TRUE(render(engine, gen_inv, px_clean));
    double mean_clean = mean_r(px_clean);
    std::cout << "WITHOUT dead, invert: mean R = " << mean_clean << std::endl;

    // --- With dead, switch to invert ---
    Graph g_dead = build_graph_with_dead();
    uint64_t gen_dead = engine.set_graph(g_dead);
    ASSERT_NE(gen_dead, 0u) << engine.last_error();
    set_params(engine);

    uint64_t gen_inv2 = engine.set_active_node(4);
    ASSERT_NE(gen_inv2, 0u) << engine.last_error();
    std::vector<float> px_dead;
    ASSERT_TRUE(render(engine, gen_inv2, px_dead));
    double mean_dead = mean_r(px_dead);
    std::cout << "WITH dead, invert:    mean R = " << mean_dead << std::endl;

    double diff = max_diff(px_clean, px_dead);
    std::cout << "Max pixel diff:    " << diff << std::endl;

    EXPECT_NEAR(mean_clean, mean_dead, 1e-6)
        << "Dead node changed switched output!";
    EXPECT_LT(diff, 1e-6)
        << "Dead node changed ANY pixel after switch! Max diff = " << diff;

    engine.shutdown();
}


// Test: Round-trip — render with dead, switch to invert, switch back, compare.
TEST(DeadNodeImpact, RoundTripWithDeadNode) {
    Engine engine;
    if (!init(engine, "cache_dead_node_roundtrip"))
        GTEST_SKIP() << engine.last_error();

    Graph g = build_graph_with_dead();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 0.5f});

    // Render output (node 12).
    std::vector<float> px1;
    ASSERT_TRUE(render(engine, gen, px1));
    double r1 = mean_r(px1);

    // Switch to invert (node 4), render.
    std::vector<float> px_inv;
    ASSERT_TRUE(render(engine, engine.set_active_node(4), px_inv));

    // Switch back to output (node 12), render.
    std::vector<float> px2;
    ASSERT_TRUE(render(engine, engine.set_active_node(12), px2));
    double r2 = mean_r(px2);

    double diff = max_diff(px1, px2);
    std::cout << "Round-trip with dead node: before=" << r1
              << " after=" << r2 << " diff=" << diff << std::endl;

    EXPECT_NEAR(r1, r2, 0.05)
        << "Output changed after round-trip with dead node present";
    EXPECT_LT(diff, 0.1)
        << "Pixel-level round-trip failure with dead node";

    engine.shutdown();
}
