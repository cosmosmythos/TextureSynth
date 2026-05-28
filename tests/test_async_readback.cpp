#include <gtest/gtest.h>
#include "engine/VulkanContext.hpp"
#include "engine/AsyncReadback.hpp"
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include <chrono>
#include <thread>
#include <vector>

using namespace te;
using namespace std::chrono_literals;

TEST(AsyncReadback, InitAndShutdownClean) {
    VulkanContext ctx;
    VulkanContextDesc d{}; d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    AsyncReadback ar;
    ASSERT_TRUE(ar.init(ctx, 2048, 2048));
    EXPECT_FALSE(ar.any_in_flight());
    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, NoInFlightAfterInit) {
    VulkanContext ctx;
    VulkanContextDesc d{};
    ASSERT_TRUE(ctx.init(d));
    AsyncReadback ar;
    ASSERT_TRUE(ar.init(ctx, 1024, 1024));
    EXPECT_FALSE(ar.any_in_flight());
    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, EnsureCapacityGrowsStaging) {
    VulkanContext ctx;
    VulkanContextDesc d{};
    ASSERT_TRUE(ctx.init(d));
    AsyncReadback ar;
    ASSERT_TRUE(ar.init(ctx, 256, 256));

    EXPECT_TRUE(ar.ensure_capacity(ctx, 1024, 1024));
    EXPECT_TRUE(ar.ensure_capacity(ctx, 1024, 1024));
    EXPECT_TRUE(ar.ensure_capacity(ctx, 512, 512));

    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, PollOnEmptyReturnsFalse) {
    VulkanContext ctx;
    VulkanContextDesc d{};
    ASSERT_TRUE(ctx.init(d));
    AsyncReadback ar;
    ASSERT_TRUE(ar.init(ctx, 512, 512));

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    uint64_t gen = 0;
    EXPECT_FALSE(ar.poll(ctx, px, w, h, gen));
    EXPECT_TRUE(px.empty());

    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, SyntheticPublishAndPoll) {
    VulkanContext ctx;
    VulkanContextDesc d{};
    ASSERT_TRUE(ctx.init(d));
    AsyncReadback ar;
    ASSERT_TRUE(ar.init(ctx, 64, 64));

    std::vector<float> src(8 * 8 * 4, 0.5f);
    uint64_t ticket = ar.publish_synthetic(src, 8, 8, 42);
    EXPECT_GT(ticket, 0u);

    std::vector<float> got;
    uint32_t w = 0, h = 0;
    uint64_t gen = 0;
    ASSERT_TRUE(ar.poll(ctx, got, w, h, gen));
    EXPECT_EQ(w, 8u);
    EXPECT_EQ(h, 8u);
    EXPECT_EQ(gen, 42u);
    ASSERT_EQ(got.size(), src.size());
    EXPECT_FLOAT_EQ(got.front(), 0.5f);
    EXPECT_FLOAT_EQ(got.back(),  0.5f);

    EXPECT_FALSE(ar.poll(ctx, got, w, h, gen));

    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, DrainOnIdleIsNoOp) {
    VulkanContext ctx;
    VulkanContextDesc d{};
    ASSERT_TRUE(ctx.init(d));
    AsyncReadback ar;
    ASSERT_TRUE(ar.init(ctx, 256, 256));

    EXPECT_FALSE(ar.any_in_flight());
    ar.drain(ctx);
    EXPECT_FALSE(ar.any_in_flight());

    ar.shutdown(ctx);
    ctx.shutdown();
}

namespace {

bool wait_for_pipeline(Engine& e, int max_ms = 5000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        e.poll_pending_compiles();
        if (e.has_pipeline()) return true;
        std::this_thread::sleep_for(10ms);
    }
    return false;
}

bool wait_for_readback(Engine& e,
                       std::vector<float>& pixels,
                       uint32_t& w, uint32_t& h,
                       uint64_t& gen,
                       int max_ms = 5000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (e.async_readback().poll(e.ctx(), pixels, w, h, gen)) return true;
        std::this_thread::sleep_for(5ms);
    }
    return false;
}

bool init_engine_with_nodes(Engine& e) {
    return e.init(VK_NULL_HANDLE, nullptr, 0,
                  true,
                  "test_shader_cache",
                  "shader_assets/nodes",
                  "shader_assets/glsl");
}

bool build_trivial_graph(const Engine& e, Graph& g_out) {
    for (const auto& [id, type] : e.node_library().all()) {
        if (type.inputs.empty()) {
            g_out.nodes.push_back({1, id});
            g_out.output_node = 1;
            return true;
        }
    }
    return false;
}

} // namespace

TEST(AsyncReadback, SubmitAndPollProducesPixels) {
    Engine engine;
    ASSERT_TRUE(init_engine_with_nodes(engine));

    Graph g;
    if (!build_trivial_graph(engine, g)) {
        GTEST_SKIP() << "No zero-input node available in node library.";
    }

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine)) << "pipeline compile timed out";

    PushConstants pc{};
    pc.resolution_x = 256;
    pc.resolution_y = 256;
    pc.seed = 7;
    pc.time = 0.0f;

    engine.set_resolution(256, 256);

    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);

    std::vector<float> pixels;
    uint32_t w = 0, h = 0;
    uint64_t out_gen = 0;
    ASSERT_TRUE(wait_for_readback(engine, pixels, w, h, out_gen));

    EXPECT_EQ(w, 256u);
    EXPECT_EQ(h, 256u);
    EXPECT_EQ(out_gen, gen);
    EXPECT_EQ(pixels.size(), 256u * 256u * 4u);

    engine.shutdown();
}

TEST(AsyncReadback, RingFillsAndDrains) {
    Engine engine;
    ASSERT_TRUE(init_engine_with_nodes(engine));

    Graph g;
    if (!build_trivial_graph(engine, g)) {
        GTEST_SKIP() << "No zero-input node available.";
    }
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u);
    ASSERT_TRUE(wait_for_pipeline(engine));

    PushConstants pc{};
    pc.resolution_x = 128;
    pc.resolution_y = 128;

    int submitted = 0;
    int rejected  = 0;
    for (int i = 0; i < 8; ++i) {
        pc.seed = static_cast<uint32_t>(i + 1);
        engine.mark_node_dirty(1);
        uint64_t t = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
        if (t == 0) ++rejected; else ++submitted;
    }
    EXPECT_GE(submitted, 1) << "at least one slot should accept a job";
    EXPECT_LE(submitted, (int)AsyncReadback::SLOT_COUNT)
        << "ring should not accept more than SLOT_COUNT concurrent jobs";

    engine.async_readback().drain(engine.ctx());
    EXPECT_FALSE(engine.async_readback().any_in_flight());

    engine.shutdown();
}

TEST(AsyncReadback, GenerationMismatchAfterRebuild) {
    Engine engine;
    ASSERT_TRUE(init_engine_with_nodes(engine));

    Graph g;
    if (!build_trivial_graph(engine, g)) {
        GTEST_SKIP();
    }

    uint64_t gen1 = engine.set_graph(g);
    ASSERT_NE(gen1, 0u);
    ASSERT_TRUE(wait_for_pipeline(engine));
    EXPECT_TRUE(engine.is_generation_ready(gen1));

    uint64_t gen2 = engine.set_graph(g);
    ASSERT_NE(gen2, 0u);
    EXPECT_NE(gen1, gen2);
    ASSERT_TRUE(wait_for_pipeline(engine));
    EXPECT_TRUE(engine.is_generation_ready(gen2));
    EXPECT_FALSE(engine.is_generation_ready(gen1)) << "stale generation must not look ready";

    engine.shutdown();
}

TEST(AsyncReadback, DirtySkipRepublishesLastFrame) {
    Engine engine;
    ASSERT_TRUE(init_engine_with_nodes(engine));

    Graph g;
    if (!build_trivial_graph(engine, g)) {
        GTEST_SKIP();
    }
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u);
    ASSERT_TRUE(wait_for_pipeline(engine));

    PushConstants pc{};
    pc.resolution_x = 128;
    pc.resolution_y = 128;
    pc.seed = 1;

    uint64_t t1 = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(t1, 0u);

    std::vector<float> px1;
    uint32_t w1 = 0, h1 = 0;
    uint64_t g1 = 0;
    ASSERT_TRUE(wait_for_readback(engine, px1, w1, h1, g1));

    engine.stash_last_presented(px1, w1, h1, g1);

    EXPECT_FALSE(engine.any_pass_dirty());
    EXPECT_TRUE(engine.has_presented_frame());

    uint64_t t2 = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    EXPECT_NE(t2, 0u);

    std::vector<float> px2;
    uint32_t w2 = 0, h2 = 0;
    uint64_t g2 = 0;
    ASSERT_TRUE(wait_for_readback(engine, px2, w2, h2, g2));
    EXPECT_EQ(w2, w1);
    EXPECT_EQ(h2, h1);
    EXPECT_EQ(px2.size(), px1.size());
    EXPECT_EQ(px2, px1);

    engine.shutdown();
}
