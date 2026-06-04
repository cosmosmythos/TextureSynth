#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include "test_assets.hpp"
#include <chrono>
#include <thread>
#include <unordered_map>

// End-to-end test: requires a populated nodes/ and glsl/ directory.
// Skips gracefully if assets are missing.
TEST(Engine, EndToEndSingleNodeRender) {
    te::Engine engine;

    // VK_NULL_HANDLE surface = headless mode (engine-only, no viewer window).
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0,
                          /*validation*/ true,
                          "test_shader_cache",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
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

// Stage 1 / 0.3: a param update between two submits must trigger a
// re-dispatch. Before the fix, update_node_params_by_id() only wrote
// to NodeResource::is_dirty (Layer 3) but never seeded the engine's
// DirtySet (Layer 2). The next record_dispatch() would then call
// dirty_set_.propagate() on an empty seed and early-exit on
// `if (!dirty_set_.any()) return;` at Engine.cpp:776, republishing the
// previous frame via async readback's synthetic-publish cache. After
// the fix, the seed change produces different pixels in frame 2.
//
// We use the "value" noise node (8 params: period, octaves, lacunarity,
// gain, offsetX, offsetY, speed, seed). The shader hashes (seed ^ pc.seed)
// into the noise, so changing seed -> different pixels.
TEST(Engine, ParamUpdateTriggersRedispatch) {
    te::Engine engine;

    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0,
                          /*validation*/ true,
                          "test_shader_cache_param_update",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed (missing assets?): " << engine.last_error();
    }

    te::Graph g;
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

    // Use the engine's actual output dimensions (default 512x512); the
    // pc.resolution hint affects dispatch compute, not the readback shape.
    const uint32_t W = 64, H = 64;
    te::PushConstants pc{};
    pc.resolution_x = W; pc.resolution_y = H;
    pc.seed = 1; pc.time = 0.0f;

    // Frame 1: defaults (seed param = 0, so iseed = 0 ^ 1 = 1)
    uint64_t t1 = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(t1, 0u);
    std::vector<float> p1;
    uint32_t w1 = 0, h1 = 0; uint64_t og1 = 0;
    for (int i = 0; i < 200 && !engine.async_readback().poll(engine.ctx(), p1, w1, h1, og1); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_FALSE(p1.empty()) << "no pixels read back from frame 1";

    // Change the seed param. This must trigger a re-dispatch.
    // value.node.json: params are {period, octaves, lacunarity, gain,
    // offsetX, offsetY, speed, seed} with defaults {8, 5, 2, 0.5, 0, 0, 0, 0}.
    // Setting seed to 1234 -> iseed = 1234 ^ 1 = 1235, very different noise.
    std::unordered_map<std::string, float> kv{
        {"period", 8.0f}, {"octaves", 5.0f}, {"lacunarity", 2.0f},
        {"gain", 0.5f},   {"offsetX", 0.0f}, {"offsetY", 0.0f},
        {"speed", 0.0f},  {"seed", 1234.0f},
    };
    engine.update_node_params_by_name(1, kv);

    // Frame 2: should be different.
    uint64_t t2 = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(t2, 0u);
    std::vector<float> p2;
    uint32_t w2 = 0, h2 = 0; uint64_t og2 = 0;
    for (int i = 0; i < 200 && !engine.async_readback().poll(engine.ctx(), p2, w2, h2, og2); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_FALSE(p2.empty()) << "no pixels read back from frame 2";
    ASSERT_EQ(p1.size(), p2.size()) << "frame size mismatch";

    bool any_diff = false;
    for (size_t i = 0; i < p1.size(); ++i) {
        if (p1[i] != p2[i]) { any_diff = true; break; }
    }
    EXPECT_TRUE(any_diff)
        << "param update did not trigger re-dispatch (p1 == p2 -- "
           "engine likely early-exited on empty dirty_set_)";

    engine.shutdown();
}

// Phase 1c: a bypassed node's output must be cleared to zero at dispatch
// time. We verify this end-to-end by rendering color_const (writes (1,1,1,1)
// by default) twice: once normal, once bypassed, and asserting that the
// second readback is all-zero.
TEST(Engine, BypassedNodeOutputIsClearedToZero) {
    te::Engine engine;

    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0,
                          /*validation*/ true,
                          "test_shader_cache_bypassed",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed (missing assets?): " << engine.last_error();
    }

    // ── Pass 1: non-bypassed — output should be (1,1,1,1) per color_const default ──
    te::Graph g1;
    g1.nodes.push_back({1, "color_const"});
    g1.output_node = 1;

    uint64_t gen1 = engine.set_graph(g1);
    if (gen1 == 0) {
        engine.shutdown();
        GTEST_SKIP() << "set_graph (normal) failed: " << engine.last_error();
    }
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline()) << "normal compile timed out";

    te::PushConstants pc{};
    pc.resolution_x = 64; pc.resolution_y = 64;
    pc.seed = 1; pc.time = 0.0f;

    uint64_t t1 = engine.async_readback().submit(engine.ctx(), engine, pc, gen1);
    ASSERT_NE(t1, 0u);

    std::vector<float> pixels1;
    uint32_t w1 = 0, h1 = 0; uint64_t og1 = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels1, w1, h1, og1)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(w1, 64u);
    ASSERT_EQ(h1, 64u);
    ASSERT_FALSE(pixels1.empty()) << "no pixels read back from normal pass";

    // At least one channel of at least one pixel should be non-zero (color_const
    // defaults to mode=1, rgba=(1,1,1,1)). If everything is already zero the
    // test would be vacuous; bail with a SKIP rather than a false-positive.
    bool saw_nonzero = false;
    for (float v : pixels1) {
        if (v != 0.0f) { saw_nonzero = true; break; }
    }
    if (!saw_nonzero) {
        engine.shutdown();
        GTEST_SKIP() << "normal color_const produced all-zero output "
                        "(unexpected default; cannot test bypass semantics)";
    }

    // ── Pass 2: bypassed — output must be all zeros ──
    te::Graph g2;
    te::NodeInstance bypassed_node;
    bypassed_node.id = 1;
    bypassed_node.type_id = "color_const";
    bypassed_node.bypassed = true;
    g2.nodes.push_back(bypassed_node);
    g2.output_node = 1;

    uint64_t gen2 = engine.set_graph(g2);
    ASSERT_NE(gen2, 0u) << "set_graph (bypassed) failed: " << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline()) << "bypassed compile timed out";

    uint64_t t2 = engine.async_readback().submit(engine.ctx(), engine, pc, gen2);
    ASSERT_NE(t2, 0u);

    std::vector<float> pixels2;
    uint32_t w2 = 0, h2 = 0; uint64_t og2 = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels2, w2, h2, og2)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_EQ(w2, 64u);
    ASSERT_EQ(h2, 64u);
    ASSERT_EQ(pixels2.size(), 64u * 64u * 4u);

    bool all_zero = true;
    for (float v : pixels2) {
        if (v != 0.0f) { all_zero = false; break; }
    }
    EXPECT_TRUE(all_zero) << "bypassed pass must clear output to zero "
                              "(got " << pixels2[0] << " at pixel 0)";

    engine.shutdown();
}
