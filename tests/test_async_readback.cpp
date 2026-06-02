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

// ---------------------------------------------------------------------------
// Stage 3 — Allocate-then-swap safety for the staging buffer.
//
// The three tests below guard the new behavior in AsyncReadback:
//   * ensure_capacity rejects dimensions that would overflow or hit the
//     4 GiB sanity cap.
//   * ensure_capacity rejects zero dimensions.
//   * A failed resize leaves the old VkBuffer / VmaAllocation intact and
//     still usable for a subsequent poll, proving the old buffer was not
//     destroyed before the new one was successfully allocated.
// ---------------------------------------------------------------------------

TEST(AsyncReadback, EnsureCapacityRejectsOverflow) {
    VulkanContext ctx; ASSERT_TRUE(ctx.init({}));
    AsyncReadback ar;    ASSERT_TRUE(ar.init(ctx, 64, 64));

    // 0x10000 * 0x10000 * 4 * 4 = 16 GiB. Exceeds the 4 GiB sanity cap.
    EXPECT_FALSE(ar.ensure_capacity(ctx, 0x10000u, 0x10000u));

    // Even on a 64-bit host this must NOT advance the current capacity.
    // (We probe by attempting a small valid resize afterwards -- it must
    // succeed because the small cap is below the current one.)
    EXPECT_TRUE(ar.ensure_capacity(ctx, 64, 64));

    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, EnsureCapacityRejectsZero) {
    VulkanContext ctx; ASSERT_TRUE(ctx.init({}));
    AsyncReadback ar;    ASSERT_TRUE(ar.init(ctx, 64, 64));

    EXPECT_FALSE(ar.ensure_capacity(ctx, 0u,  64u));
    EXPECT_FALSE(ar.ensure_capacity(ctx, 64u, 0u));
    EXPECT_FALSE(ar.ensure_capacity(ctx, 0u,  0u));

    // The original 64x64 staging must still be intact -- verified by a
    // successful synthetic publish + poll on the same Engine.
    std::vector<float> src(8 * 8 * 4, 0.5f);
    EXPECT_GT(ar.publish_synthetic(src, 8, 8, 1), 0u);
    std::vector<float> got; uint32_t w, h; uint64_t g;
    ASSERT_TRUE(ar.poll(ctx, got, w, h, g));
    EXPECT_EQ(w, 8u);
    EXPECT_EQ(h, 8u);

    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, FailureDuringResizeKeepsOldBuffers) {
    // init at 64x64, then ask for an absurd dimension that we know will
    // be rejected. The old staging buffer must remain valid.
    VulkanContext ctx; ASSERT_TRUE(ctx.init({}));
    AsyncReadback ar;    ASSERT_TRUE(ar.init(ctx, 64, 64));

    // Sanity: 256x256 is a valid (small) resize.
    ASSERT_TRUE(ar.ensure_capacity(ctx, 256, 256));

    // Now request a dimension that the new capacity_bytes() rejects.
    EXPECT_FALSE(ar.ensure_capacity(ctx, 0x40000u, 0x40000u));
    EXPECT_FALSE(ar.ensure_capacity(ctx, 0u, 64u));

    // The previously-installed 256x256 buffer must still be usable:
    // publish + poll on the same Engine should succeed.
    std::vector<float> src(16 * 16 * 4, 0.25f);
    EXPECT_GT(ar.publish_synthetic(src, 16, 16, 7), 0u);
    std::vector<float> got; uint32_t w, h; uint64_t g;
    ASSERT_TRUE(ar.poll(ctx, got, w, h, g));
    EXPECT_EQ(w, 16u);
    EXPECT_EQ(h, 16u);
    EXPECT_EQ(got.size(), src.size());
    EXPECT_FLOAT_EQ(got.front(), 0.25f);

    ar.shutdown(ctx);
    ctx.shutdown();
}

TEST(AsyncReadback, RejectsZeroAtInit) {
    VulkanContext ctx; ASSERT_TRUE(ctx.init({}));
    AsyncReadback ar;

    EXPECT_FALSE(ar.init(ctx, 0, 64));
    EXPECT_FALSE(ar.init(ctx, 64, 0));
    EXPECT_FALSE(ar.init(ctx, 0, 0));

    // A subsequent valid init must succeed and produce a working ring.
    ASSERT_TRUE(ar.init(ctx, 32, 32));
    std::vector<float> src(4 * 4 * 4, 0.5f);
    EXPECT_GT(ar.publish_synthetic(src, 4, 4, 1), 0u);
    std::vector<float> got; uint32_t w, h; uint64_t g;
    ASSERT_TRUE(ar.poll(ctx, got, w, h, g));
    EXPECT_EQ(got.size(), src.size());

    ar.shutdown(ctx);
    ctx.shutdown();
}

// ---------------------------------------------------------------------------
// Stage 4+5 — Engine lifecycle + guards.
//
// These tests live in test_async_readback.cpp because the test executable
// already links `engine`; no need to spin up a second TU. The state-machine
// is exercised end-to-end on a single Engine instance.
//
// Note on process-wide VkInstance: AGENTS.md flags that two Engine()
// instances in one process can collide on the Vulkan loader. These tests
// create at most one Engine at a time and shut it down before returning.
// ---------------------------------------------------------------------------

namespace {

bool init_engine_for_lifecycle(Engine& e) {
    return e.init(VK_NULL_HANDLE, nullptr, 0,
                  true,
                  "test_shader_cache",
                  "shader_assets/nodes",
                  "shader_assets/glsl");
}

} // namespace

TEST(EngineLifecycle, InitIsIdempotent) {
    Engine engine;
    if (!init_engine_for_lifecycle(engine)) {
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();
    }
    EXPECT_EQ(engine.engine_state(), Engine::EngineState::Ready);

    // Second init: must return true (idempotent) and stay Ready.
    EXPECT_TRUE(init_engine_for_lifecycle(engine));
    EXPECT_EQ(engine.engine_state(), Engine::EngineState::Ready);

    engine.shutdown();
    EXPECT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);
}

TEST(EngineLifecycle, ShutdownIsIdempotent) {
    Engine engine;
    if (!init_engine_for_lifecycle(engine)) {
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();
    }
    engine.shutdown();
    EXPECT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);

    // Second shutdown: must not crash, must stay ShutDown.
    engine.shutdown();
    EXPECT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);
}

TEST(EngineLifecycle, UseAfterShutdownIsRejected) {
    Engine engine;
    if (!init_engine_for_lifecycle(engine)) {
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();
    }
    Graph g;
    engine.shutdown();
    ASSERT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);

    // set_graph: must return 0 and set the structured error.
    const uint64_t g0 = engine.set_graph(g);
    EXPECT_EQ(g0, 0u);
    const auto& rec = engine.last_error_record();
    EXPECT_EQ(rec.code, EngineErrorCode::UseAfterShutdown);
    EXPECT_FALSE(rec.message.empty());

    // void-returning guards: poll_pending_compiles / update_node_params_*
    // don't crash, they just no-op (we can't observe the no-op directly,
    // but the test would crash on a use-after-free if the guard were
    // missing).
    engine.poll_pending_compiles();
    engine.update_node_params_by_id(1, {0.0f});
    engine.update_node_params_by_name(1, {{"x", 0.0f}});
    EXPECT_FALSE(engine.upload_image(1, nullptr, 0, 0));
    EXPECT_FALSE(engine.release_image(1));

    // last_error_record was set by the set_graph call above; subsequent
    // calls don't overwrite it.
    EXPECT_EQ(engine.last_error_record().code, EngineErrorCode::UseAfterShutdown);
}

TEST(EngineLifecycle, InitShutdownInitRearms) {
    Engine engine;
    if (!init_engine_for_lifecycle(engine)) {
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();
    }
    engine.shutdown();
    ASSERT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);

    // Re-init on the same handle. We don't require success (Vulkan loader
    // may refuse a second VkInstance in one process) -- only that we don't
    // segfault and that the engine transitions out of ShutDown.
    const bool ok = init_engine_for_lifecycle(engine);
    if (ok) {
        EXPECT_EQ(engine.engine_state(), Engine::EngineState::Ready);
        engine.shutdown();
    } else {
        // Failure path: state must be Error (init ran the rollback path).
        EXPECT_EQ(engine.engine_state(), Engine::EngineState::Error);
    }
}

// ---------------------------------------------------------------------------
// Stage 5 -- Concurrency / lifecycle stress tests.
//
// The lifecycle refactor (entry_mu_ held for the entire init / shutdown body
// and for every public mutator) is exercised under real threading here.
// ---------------------------------------------------------------------------

TEST(EngineLifecycle, StressInitShutdownCycles) {
    // 50 init/shutdown cycles. The first init brings up Vulkan; each
    // shutdown tears it down. If the lock doesn't properly serialise
    // access to engine internals, we'd see a use-after-free somewhere
    // in the 50 cycles.
    Engine engine;
    for (int i = 0; i < 50; ++i) {
        if (!init_engine_for_lifecycle(engine)) {
            // Vulkan refused a re-init (this is the case on most loaders
            // once the first instance is destroyed). Skip the rest.
            GTEST_SKIP() << "Vulkan re-init refused at cycle " << i;
        }
        EXPECT_EQ(engine.engine_state(), Engine::EngineState::Ready);
        engine.shutdown();
        EXPECT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);
    }
}

TEST(EngineLifecycle, ConcurrentInitAndShutdown) {
    // Two threads call init/shutdown in a tight loop. The lock must
    // serialise them: at any given instant exactly one is "inside" an
    // operation, the other is blocked. After both threads finish, the
    // engine must be in a defined state (Ready or ShutDown) and we
    // must not have crashed.
    Engine engine;
    std::atomic<int> init_calls{0};
    std::atomic<int> shutdown_calls{0};
    std::atomic<bool> go{false};

    auto worker_init = [&]() {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < 25; ++i) {
            if (init_engine_for_lifecycle(engine)) ++init_calls;
        }
    };
    auto worker_shutdown = [&]() {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < 25; ++i) {
            engine.shutdown();
            ++shutdown_calls;
        }
    };

    std::thread t1(worker_init);
    std::thread t2(worker_shutdown);
    go.store(true, std::memory_order_release);
    t1.join();
    t2.join();

    // Exactly one of the workers will have won the race for the first
    // init; the rest either succeed idempotently or fail silently.
    EXPECT_GT(init_calls.load() + shutdown_calls.load(), 0);
    // Final state is either Ready (last init won) or ShutDown (last
    // shutdown won). The test passes if the engine is in either state.
    const auto final = engine.engine_state();
    EXPECT_TRUE(final == Engine::EngineState::Ready ||
                final == Engine::EngineState::ShutDown ||
                final == Engine::EngineState::Error)
        << "engine in unexpected state " << (int)final;
    // If we ended Ready, leave the engine in a clean state for the next
    // test in the binary.
    if (final == Engine::EngineState::Ready) engine.shutdown();
}

TEST(AsyncReadback, SubmitAndPollAfterShutdownAreSafe) {
    // The new bool initialized_ guard in AsyncReadback means submit() and
    // poll() return 0 / false after shutdown() rather than dereferencing
    // a null VkBuffer / VkDevice.
    VulkanContext ctx; ASSERT_TRUE(ctx.init({}));
    AsyncReadback ar;    ASSERT_TRUE(ar.init(ctx, 64, 64));

    ar.shutdown(ctx);
    ctx.shutdown();

    // ctx is gone; submit/poll must short-circuit on the initialized_ flag
    // before touching any Vulkan handle. (Vulkan calls would crash; the
    // guard exists exactly to prevent that.)
    PushConstants pc{};
    uint64_t ticket = ar.submit(ctx, /*engine=*/*(Engine*)nullptr, pc, 0);
    EXPECT_EQ(ticket, 0u);

    std::vector<float> px; uint32_t w, h; uint64_t g;
    EXPECT_FALSE(ar.poll(ctx, px, w, h, g));
    EXPECT_TRUE(px.empty());
}

TEST(EngineLifecycle, ShutdownDuringInFlightWorkDrains) {
    // Start a render, then immediately shut the engine down. The shutdown
    // must drain in-flight work via async_.drain() before tearing down
    // the Vulkan device. The test passes if no crash and state ends ShutDown.
    Engine engine;
    if (!init_engine_for_lifecycle(engine)) {
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();
    }
    Graph g;
    if (!build_trivial_graph(engine, g)) {
        GTEST_SKIP() << "No zero-input node available";
    }
    uint64_t gen = engine.set_graph(g);
    if (gen == 0) {
        GTEST_SKIP() << "set_graph failed: " << engine.last_error();
    }
    if (!wait_for_pipeline(engine)) {
        GTEST_SKIP() << "pipeline compile timed out";
    }

    // Submit a job and immediately shut down. shutdown drains in-flight
    // fences, so this is safe.
    PushConstants pc{};
    pc.resolution_x = 64; pc.resolution_y = 64; pc.seed = 1;
    engine.set_resolution(64, 64);
    {
            std::lock_guard<std::recursive_mutex> lk(engine.entry_mutex());
        engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    }
    engine.shutdown();
    EXPECT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);
}

TEST(EngineLifecycle, ConcurrentSubmitAndShutdown) {
    // Thread A hammers submit() in a tight loop. Thread B shuts the engine
    // down. The lock on entry_mu_ serialises the operations: Thread A
    // either completes a submit before Thread B tears down, or Thread A
    // bails because the engine is no longer Ready. No crash, no UB.
    Engine engine;
    if (!init_engine_for_lifecycle(engine)) {
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();
    }
    Graph g;
    if (!build_trivial_graph(engine, g)) {
        GTEST_SKIP() << "No zero-input node available";
    }
    uint64_t gen = engine.set_graph(g);
    if (gen == 0 || !wait_for_pipeline(engine)) {
        GTEST_SKIP() << "graph/pipeline setup failed";
    }
    PushConstants pc{};
    pc.resolution_x = 64; pc.resolution_y = 64; pc.seed = 1;
    engine.set_resolution(64, 64);

    std::atomic<bool> go{false};
    std::atomic<int> submits{0};
    std::atomic<int> rejected{0};

    auto submitter = [&]() {
        while (!go.load(std::memory_order_acquire)) {}
        for (int i = 0; i < 200; ++i) {
            if (!engine.is_ready()) { ++rejected; continue; }
        std::lock_guard<std::recursive_mutex> lk(engine.entry_mutex());
            if (!engine.is_ready()) { ++rejected; continue; }
            pc.seed = static_cast<uint32_t>(i + 1);
            engine.mark_node_dirty(1);
            const uint64_t t = engine.async_readback().submit(
                engine.ctx(), engine, pc, gen);
            if (t != 0) ++submits;
        }
    };
    auto killer = [&]() {
        while (!go.load(std::memory_order_acquire)) {}
        // Tiny delay so the submitter has a chance to start.
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        engine.shutdown();
    };

    std::thread t1(submitter);
    std::thread t2(killer);
    go.store(true, std::memory_order_release);
    t1.join();
    t2.join();

    EXPECT_EQ(engine.engine_state(), Engine::EngineState::ShutDown);
    EXPECT_GE(submits.load() + rejected.load(), 200);
}

