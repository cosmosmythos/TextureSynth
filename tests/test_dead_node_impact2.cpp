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
#include <algorithm>
#include <iostream>
#include <vector>

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

double mean_r(const std::vector<float>& px) {
    double s = 0; size_t n = px.size() / 4;
    for (size_t i = 0; i + 3 < px.size(); i += 4) s += px[i];
    return n ? s / n : 0;
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
    g.nodes.push_back({13, "blend"}); // dead

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
    g.connections.push_back({6, 0, 13, 1}); // dead link
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


// CRITICAL: Round-trip WITHOUT dead node (12 nodes) — is round-trip itself broken?
TEST(DeadNodeImpact, RoundTrip_NoDead_12Nodes) {
    Engine engine;
    if (!init(engine, "cache_rt_nodead"))
        GTEST_SKIP() << engine.last_error();

    Graph g = build_13node();
    g.nodes.pop_back(); // remove node 13
    // Remove dead link 6->13
    g.connections.erase(
        std::remove_if(g.connections.begin(), g.connections.end(),
            [](const Connection& c){ return c.dst_node == 13; }),
        g.connections.end());
    g.output_node = 12;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);

    // Render output (node 12).
    std::vector<float> px1;
    ASSERT_TRUE(render(engine, gen, px1));
    double r1 = mean_r(px1);
    std::cout << "12-node round-trip: before=" << r1 << std::endl;

    // Switch to invert (node 4).
    std::vector<float> px_inv;
    ASSERT_TRUE(render(engine, engine.set_active_node(4), px_inv));
    std::cout << "  invert mean R = " << mean_r(px_inv) << std::endl;

    // Switch back to output (node 12).
    std::vector<float> px2;
    ASSERT_TRUE(render(engine, engine.set_active_node(12), px2));
    double r2 = mean_r(px2);
    std::cout << "  after round-trip mean R = " << r2 << std::endl;

    double diff = max_diff(px1, px2);
    std::cout << "  max pixel diff = " << diff << std::endl;

    EXPECT_NEAR(r1, r2, 0.05)
        << "12-node round-trip FAILED (no dead node)";
    EXPECT_LT(diff, 0.1)
        << "12-node pixel round-trip FAILED";

    engine.shutdown();
}


// Round-trip WITH dead node (13 nodes) — is dead node the cause?
TEST(DeadNodeImpact, RoundTrip_WithDead_13Nodes) {
    Engine engine;
    if (!init(engine, "cache_rt_dead"))
        GTEST_SKIP() << engine.last_error();

    Graph g = build_13node();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);

    std::vector<float> px1;
    ASSERT_TRUE(render(engine, gen, px1));
    double r1 = mean_r(px1);
    std::cout << "13-node round-trip: before=" << r1 << std::endl;

    std::vector<float> px_inv;
    ASSERT_TRUE(render(engine, engine.set_active_node(4), px_inv));
    std::cout << "  invert mean R = " << mean_r(px_inv) << std::endl;

    std::vector<float> px2;
    ASSERT_TRUE(render(engine, engine.set_active_node(12), px2));
    double r2 = mean_r(px2);
    std::cout << "  after round-trip mean R = " << r2 << std::endl;

    double diff = max_diff(px1, px2);
    std::cout << "  max pixel diff = " << diff << std::endl;

    EXPECT_NEAR(r1, r2, 0.05)
        << "13-node round-trip FAILED (with dead node)";
    EXPECT_LT(diff, 0.1)
        << "13-node pixel round-trip FAILED";

    engine.shutdown();
}


// Double render WITHOUT switching — does second render match first?
TEST(DeadNodeImpact, DoubleRender_NoSwitch) {
    Engine engine;
    if (!init(engine, "cache_double_render"))
        GTEST_SKIP() << engine.last_error();

    Graph g = build_13node();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    set_params(engine);

    std::vector<float> px1;
    ASSERT_TRUE(render(engine, gen, px1));
    double r1 = mean_r(px1);

    std::vector<float> px2;
    ASSERT_TRUE(render(engine, gen, px2));
    double r2 = mean_r(px2);

    double diff = max_diff(px1, px2);
    std::cout << "Double render: r1=" << r1 << " r2=" << r2 << " diff=" << diff << std::endl;

    EXPECT_NEAR(r1, r2, 0.01)
        << "Double render without switching changed output!";
    EXPECT_LT(diff, 0.01)
        << "Double render pixel diff too large";

    engine.shutdown();
}
