#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "test_assets.hpp"
#include <thread>
#include <chrono>

using namespace te;

namespace {

bool init_engine(Engine& engine, const char* cache_name) {
    return engine.init(VK_NULL_HANDLE, nullptr, 0, true, cache_name,
                       find_test_nodes_dir().c_str(),
                       find_test_glsl_dir().c_str());
}

bool wait_for_pipeline(Engine& engine, int timeout_ms = 2000) {
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        engine.poll_pending_compiles();
        if (engine.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return engine.has_pipeline();
}

bool wait_for_readback(Engine& engine, std::vector<float>& pixels,
                       uint32_t& w, uint32_t& h, int timeout_ms = 2000) {
    PushConstants pc{};
    pc.resolution_x = 64; pc.resolution_y = 64;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t gen = engine.compile_generation();
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    if (ticket == 0) return false;
    uint64_t og = 0;
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

} // anonymous namespace

TEST(FusedActivation, ProducesChains) {
    Engine engine;
    bool ok = init_engine(engine, "test_fused_chains");
    if (!ok) GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    auto stats = engine.resources().get_vma_stats(engine.ctx());
    EXPECT_GT(stats.node_resource_count, 0u);
    EXPECT_TRUE(engine.has_pipeline());
}

TEST(FusedActivation, SetActiveNodeSwitchesPreview) {
    Engine engine;
    bool ok = init_engine(engine, "test_fused_active_node");
    if (!ok) GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> pixels;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback(engine, pixels, w, h));
    EXPECT_GT(w, 0u);
    EXPECT_GT(h, 0u);

    double sum = 0;
    for (size_t i = 0; i + 3 < pixels.size(); i += 4)
        sum += (pixels[i] + pixels[i+1] + pixels[i+2]) / 3.0;
    EXPECT_GT(sum, 0.0) << "invert output should not be all-black";

    uint64_t gen2 = engine.set_active_node(1);
    ASSERT_NE(gen2, 0u);
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> px2;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback(engine, px2, w2, h2));
    double sum2 = 0;
    for (size_t i = 0; i + 3 < px2.size(); i += 4)
        sum2 += (px2[i] + px2[i+1] + px2[i+2]) / 3.0;
    EXPECT_GT(sum2, 0.0) << "perlin output should not be all-black";
}

TEST(FusedActivation, ParamUpdateRedispatches) {
    Engine engine;
    bool ok = init_engine(engine, "test_fused_param_update");
    if (!ok) GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "value"});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> pixels;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback(engine, pixels, w, h));
    double sum1 = 0;
    for (size_t i = 0; i + 3 < pixels.size(); i += 4)
        sum1 += pixels[i];

    engine.update_node_params_by_id(1, {1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    engine.poll_pending_compiles();

    std::vector<float> px2;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback(engine, px2, w2, h2));
    double sum2 = 0;
    for (size_t i = 0; i + 3 < px2.size(); i += 4)
        sum2 += px2[i];

    EXPECT_NE(sum1, sum2) << "param change should alter output";
}

TEST(FusedActivation, BakeProducesPixels) {
    Engine engine;
    bool ok = init_engine(engine, "test_fused_bake");
    if (!ok) GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    auto baked = engine.bake();
    EXPECT_FALSE(baked.empty()) << "bake should produce at least one target";
    if (!baked.empty()) {
        EXPECT_GT(baked[0].width, 0u);
        EXPECT_GT(baked[0].height, 0u);
        double sum = 0;
        for (size_t i = 0; i + 3 < baked[0].pixels.size(); i += 4)
            sum += (baked[0].pixels[i] + baked[0].pixels[i+1] + baked[0].pixels[i+2]) / 3.0;
        EXPECT_GT(sum, 0.0) << "baked pixels should not be all-black";
    }
}
