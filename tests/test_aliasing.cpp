#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include "test_assets.hpp"
#include <chrono>
#include <thread>

using namespace te;

// ===========================================================================
// Aliasing integration tests — verify fused-path resource allocation.
// ===========================================================================

TEST(Aliasing, SingleNodeNoAliasingStillRenders) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_single",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "", false, false});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto stats = engine.resources().get_vma_stats(engine.ctx());
    EXPECT_EQ(stats.node_resource_count, 1u);
    double eff = stats.aliasing_efficiency();
    EXPECT_GE(eff, 1.0 - 0.01) << "single-resource graph: efficiency should be ~1.0";

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_GT(pixels.size(), 0u);
    engine.shutdown();
}

TEST(Aliasing, FusedGraphSingleOutputImage) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_fused",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }

    // Diamond: perlin -> invert -> blend, perlin -> invert -> blend
    // Fused path allocates ONE VkImage for the final output.
    Graph g;
    g.nodes.push_back({1, "perlin",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({2, "invert",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({3, "perlin",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({4, "invert",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({5, "blend",    ChannelFormat::RGBA, "", false, false});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({2, 0, 5, 0});
    g.connections.push_back({4, 0, 5, 1});
    g.output_node = 5;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << "set_graph failed: " << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto stats = engine.resources().get_vma_stats(engine.ctx());
    EXPECT_EQ(stats.node_resource_count, 1u)
        << "fused path should allocate 1 VkImage (final output only)";
    EXPECT_GT(stats.node_resource_bytes, 0u);

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(w, 512u);
    EXPECT_EQ(h, 512u);
    double sum = 0.0;
    for (size_t i = 0; i < pixels.size(); i += 4) sum += pixels[i];
    EXPECT_GT(sum, 0.0) << "rendered image is all zeros";

    engine.shutdown();
}
