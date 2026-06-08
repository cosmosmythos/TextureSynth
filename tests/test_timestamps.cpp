#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include "test_assets.hpp"
#include <chrono>
#include <thread>

using namespace te;

TEST(TimestampPool, BasicCreateDestroy) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0,
                          true,
                          "test_shader_cache_ts_create",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    engine.shutdown();
}

TEST(TimestampPool, AvailableFalse) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0,
                          true,
                          "test_shader_cache_ts_avail",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) GTEST_SKIP() << "Engine init failed: " << engine.last_error();

    engine.poll_pending_compiles();

    const auto& timings = engine.last_pass_timings();
    EXPECT_TRUE(timings.empty());

    engine.shutdown();
}

TEST(TimestampPool, WriteAndRead) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0,
                          true,
                          "test_shader_cache_ts_write",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) GTEST_SKIP() << "Engine init failed: " << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "value"});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    if (gen == 0) {
        engine.shutdown();
        GTEST_SKIP() << "set_graph failed: " << engine.last_error();
    }

    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline()) << "compile timed out";

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;

    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);

    std::vector<float> pixels;
    uint32_t w = 0, h = 0; uint64_t out_gen = 0;
    for (int i = 0; i < 200; ++i) {
        engine.poll_pending_compiles();
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, out_gen)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    const auto& timings = engine.last_pass_timings();
    EXPECT_FALSE(timings.empty());
    for (const auto& pt : timings) {
        EXPECT_TRUE(pt.available);
        EXPECT_GT(pt.duration_us, 0.0);
    }

    engine.shutdown();
}
