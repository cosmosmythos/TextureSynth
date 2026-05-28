#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include <chrono>
#include <thread>

// End-to-end test: requires a populated nodes/ and glsl/ directory.
// Skips gracefully if assets are missing.
TEST(Engine, EndToEndSingleNodeRender) {
    te::Engine engine;

    // VK_NULL_HANDLE surface = headless mode (engine-only, no viewer window).
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0,
                          /*validation*/ true,
                          "test_shader_cache",
                          "shader_assets/nodes",
                          "shader_assets/glsl");
    if (!ok) {
        GTEST_SKIP() << "Engine init failed (missing assets?): " << engine.last_error();
    }

    // Build a trivial graph: a single generator node -> output.
    // Replace "constant" with a node type that exists in your nodes/ directory.
    te::Graph g;
    g.nodes.push_back({1, "value"});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    if (gen == 0) {
        engine.shutdown();
        GTEST_SKIP() << "set_graph failed: " << engine.last_error();
    }

    // Pump the compile.
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline()) << "compile timed out";

    te::PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;

    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);

    std::vector<float> pixels;
    uint32_t w = 0, h = 0; uint64_t out_gen = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, out_gen)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    EXPECT_EQ(w, 512u);
    EXPECT_EQ(h, 512u);
    EXPECT_EQ(pixels.size(), 512u * 512u * 4u);

    engine.shutdown();
}
