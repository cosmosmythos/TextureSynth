#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/PushConstants.hpp"
#include "test_assets.hpp"
#include <vector>
#include <chrono>
#include <thread>

using namespace te;
using namespace std::chrono_literals;

namespace {

bool init_engine_for_image(Engine& e) {
    return e.init(VK_NULL_HANDLE, nullptr, 0,
                  true,
                  "test_shader_cache",
                  find_test_nodes_dir().c_str(),
                  find_test_glsl_dir().c_str());
}

bool wait_for_pipeline(Engine& e, int max_ms = 5000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        e.poll_pending_compiles();
        if (e.has_pipeline()) return true;
        std::this_thread::sleep_for(10ms);
    }
    return false;
}

// poll_pending_compiles() uses vkGetFenceStatus (non-blocking).
// Uploads may not be registered on the first call.  Retry in a loop.
void wait_for_upload(Engine& e, int max_ms = 2000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(max_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        e.poll_pending_compiles();
        std::this_thread::sleep_for(5ms);
    }
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

std::vector<float> make_solid(uint32_t w, uint32_t h,
                               float r, float g, float b, float a = 1.0f) {
    std::vector<float> px(w * h * 4);
    for (uint32_t i = 0; i < w * h; ++i) {
        px[i * 4 + 0] = r;
        px[i * 4 + 1] = g;
        px[i * 4 + 2] = b;
        px[i * 4 + 3] = a;
    }
    return px;
}

std::vector<float> make_gradient(uint32_t w, uint32_t h) {
    std::vector<float> px(w * h * 4);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t i = (y * w + x) * 4;
            px[i + 0] = float(x) / float(w - 1);
            px[i + 1] = float(y) / float(h - 1);
            px[i + 2] = 0.0f;
            px[i + 3] = 1.0f;
        }
    }
    return px;
}

Graph make_image_graph(NodeId image_node_id) {
    Graph g;
    g.nodes.push_back({image_node_id, "image"});
    g.output_node = image_node_id;
    return g;
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: Upload → poll → verify image_registry_ populated
// ---------------------------------------------------------------------------
TEST(ImageUpload, UploadAndPollRegistersImage) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    constexpr uint32_t W = 64, H = 64;
    auto pixels = make_solid(W, H, 1.0f, 0.0f, 0.0f);

    const uint64_t node_id = 1;
    ASSERT_TRUE(engine.upload_image(node_id, pixels.data(), W, H));
    wait_for_upload(engine);

    // After poll, the image should be accessible for dispatch.
    // (If it wasn't registered, set_graph + dispatch would use a dummy texture.)
    Graph g = make_image_graph(node_id);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 2: Upload → graph → dispatch → readback → verify pixels
// ---------------------------------------------------------------------------
TEST(ImageUpload, UploadGraphDispatchReadback) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    constexpr uint32_t W = 64, H = 64;
    auto pixels = make_solid(W, H, 1.0f, 0.5f, 0.25f);

    const uint64_t node_id = 1;
    ASSERT_TRUE(engine.upload_image(node_id, pixels.data(), W, H));
    wait_for_upload(engine);

    Graph g = make_image_graph(node_id);
    engine.set_resolution(W, H);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine)) << "pipeline compile timed out";

    PushConstants pc{};
    pc.resolution_x = W;
    pc.resolution_y = H;
    pc.seed = 42;
    pc.time = 0.0f;

    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u) << "readback submit failed";

    std::vector<float> out;
    uint32_t ow = 0, oh = 0;
    uint64_t out_gen = 0;
    ASSERT_TRUE(wait_for_readback(engine, out, ow, oh, out_gen));
    EXPECT_EQ(ow, W);
    EXPECT_EQ(oh, H);
    EXPECT_EQ(out.size(), W * H * 4u);

    // With default scale=1, offset=0 the image node samples the texture at UV.
    // A solid-color upload should produce a solid-color readback.
    // Check center pixel (avoids edge rounding).
    uint32_t cx = W / 2, cy = H / 2;
    uint32_t ci = (cy * W + cx) * 4;
    EXPECT_NEAR(out[ci + 0], 1.0f, 0.05f) << "R channel mismatch";
    EXPECT_NEAR(out[ci + 1], 0.5f, 0.05f) << "G channel mismatch";
    EXPECT_NEAR(out[ci + 2], 0.25f, 0.05f) << "B channel mismatch";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 3: Upload gradient → dispatch → verify spatial variation
// ---------------------------------------------------------------------------
TEST(ImageUpload, GradientUploadPreservesSpatialData) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    constexpr uint32_t W = 128, H = 128;
    auto pixels = make_gradient(W, H);

    const uint64_t node_id = 1;
    ASSERT_TRUE(engine.upload_image(node_id, pixels.data(), W, H));
    wait_for_upload(engine);

    Graph g = make_image_graph(node_id);
    engine.set_resolution(W, H);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    PushConstants pc{};
    pc.resolution_x = W;
    pc.resolution_y = H;
    pc.seed = 1;
    pc.time = 0.0f;

    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);

    std::vector<float> out;
    uint32_t ow, oh; uint64_t og;
    ASSERT_TRUE(wait_for_readback(engine, out, ow, oh, og));
    EXPECT_EQ(ow, W);
    EXPECT_EQ(oh, H);

    // Top-left should be dark (near 0,0), bottom-right should be bright (near 1,1).
    uint32_t tl_i = 0;
    uint32_t br_i = ((H - 1) * W + (W - 1)) * 4;
    float tl_lum = out[tl_i] + out[tl_i + 1] + out[tl_i + 2];
    float br_lum = out[br_i] + out[br_i + 1] + out[br_i + 2];
    EXPECT_LT(tl_lum, br_lum) << "gradient should go dark→bright top-left to bottom-right";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 4: Upload A → dispatch → upload B → dispatch → verify output changed
// ---------------------------------------------------------------------------
TEST(ImageUpload, ReuploadChangesOutput) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    constexpr uint32_t W = 32, H = 32;
    const uint64_t node_id = 1;

    // Upload red
    auto red = make_solid(W, H, 1.0f, 0.0f, 0.0f);
    ASSERT_TRUE(engine.upload_image(node_id, red.data(), W, H));
    wait_for_upload(engine);

    Graph g = make_image_graph(node_id);
    engine.set_resolution(W, H);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    PushConstants pc{};
    pc.resolution_x = W;
    pc.resolution_y = H;
    pc.seed = 1;

    uint64_t t1 = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(t1, 0u);
    std::vector<float> out1;
    uint32_t w1, h1; uint64_t g1;
    ASSERT_TRUE(wait_for_readback(engine, out1, w1, h1, g1));

    // Upload blue
    auto blue = make_solid(W, H, 0.0f, 0.0f, 1.0f);
    ASSERT_TRUE(engine.upload_image(node_id, blue.data(), W, H));
    wait_for_upload(engine);

    // Re-set graph to pick up new image binding
    gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    uint64_t t2 = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(t2, 0u);
    std::vector<float> out2;
    uint32_t w2, h2; uint64_t g2;
    ASSERT_TRUE(wait_for_readback(engine, out2, w2, h2, g2));

    // First output should be red-dominant, second should be blue-dominant.
    uint32_t ci = (H / 2 * W + W / 2) * 4;
    EXPECT_GT(out1[ci + 0], out1[ci + 2]) << "first dispatch should be red-dominant";
    EXPECT_GT(out2[ci + 2], out2[ci + 0]) << "second dispatch should be blue-dominant";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 5: Release before upload is a safe no-op (bug fix validation)
// ---------------------------------------------------------------------------
TEST(ImageUpload, ReleaseBeforeUploadIsNoop) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    // release_image on a node that was never uploaded must not error.
    EXPECT_TRUE(engine.release_image(999));

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 6: Upload → release → verify image gone → re-upload → dispatch
// ---------------------------------------------------------------------------
TEST(ImageUpload, UploadReleaseReupload) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    constexpr uint32_t W = 32, H = 32;
    const uint64_t node_id = 1;

    auto green = make_solid(W, H, 0.0f, 1.0f, 0.0f);
    ASSERT_TRUE(engine.upload_image(node_id, green.data(), W, H));
    wait_for_upload(engine);

    // Release should succeed.
    EXPECT_TRUE(engine.release_image(node_id));

    // Re-upload.
    auto yellow = make_solid(W, H, 1.0f, 1.0f, 0.0f);
    ASSERT_TRUE(engine.upload_image(node_id, yellow.data(), W, H));
    wait_for_upload(engine);

    Graph g = make_image_graph(node_id);
    engine.set_resolution(W, H);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    PushConstants pc{};
    pc.resolution_x = W;
    pc.resolution_y = H;
    pc.seed = 1;

    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);

    std::vector<float> out;
    uint32_t ow, oh; uint64_t og;
    ASSERT_TRUE(wait_for_readback(engine, out, ow, oh, og));

    uint32_t ci = (H / 2 * W + W / 2) * 4;
    EXPECT_NEAR(out[ci + 0], 1.0f, 0.05f) << "R";
    EXPECT_NEAR(out[ci + 1], 1.0f, 0.05f) << "G";
    EXPECT_NEAR(out[ci + 2], 0.0f, 0.05f) << "B";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 7: Release while upload is pending (in-flight) drains and succeeds
// ---------------------------------------------------------------------------
TEST(ImageUpload, ReleasePendingUploadDrains) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    constexpr uint32_t W = 16, H = 16;
    const uint64_t node_id = 1;

    auto px = make_solid(W, H, 0.5f, 0.5f, 0.5f);
    ASSERT_TRUE(engine.upload_image(node_id, px.data(), W, H));

    // Don't poll — release should drain the pending upload and succeed.
    EXPECT_TRUE(engine.release_image(node_id));

    // Second release (nothing left) should also be a safe no-op.
    EXPECT_TRUE(engine.release_image(node_id));

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Test 8: Build graph BEFORE upload — chain slots must not go stale.
//
// Reproduces the bug where chain_in_sampled_slots[] is snapshotted with the
// dummy (slot 0) before the image upload completes, and never updated after.
// The per-pass pe.in_sampled_slots[] IS patched by poll_completed_uploads_(),
// but the chain snapshot is not — so fused chain dispatch reads the dummy.
// ---------------------------------------------------------------------------
TEST(ImageUpload, GraphBeforeUpload_ChainSlotsNotStale) {
    Engine engine;
    if (!init_engine_for_image(engine))
        GTEST_SKIP() << "Vulkan init failed: " << engine.last_error();

    constexpr uint32_t W = 64, H = 64;
    const uint64_t image_id = 1;
    const uint64_t levels_id = 2;

    // Build graph FIRST (image not uploaded yet).
    // Image -> Levels -> Output. The fuser should put both in a chain,
    // which means chain_in_sampled_slots[0] is set at populate_chains_()
    // time. If the image hasn't been uploaded yet, this slot will be the dummy.
    Graph g;
    g.nodes.push_back({image_id, "image"});
    g.nodes.push_back({levels_id, "levels"});
    g.connections.push_back({image_id, 0, levels_id, 0});
    g.output_node = levels_id;

    engine.set_resolution(W, H);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine)) << "pipeline compile timed out";

    // NOW upload the image — after the graph (and chain) was built.
    // poll_completed_uploads_() will patch pe.in_sampled_slots[] but NOT
    // chain_in_sampled_slots[] — this is the bug we're testing.
    auto pixels = make_solid(W, H, 1.0f, 0.5f, 0.25f);
    ASSERT_TRUE(engine.upload_image(image_id, pixels.data(), W, H));
    wait_for_upload(engine);
    PushConstants pc{};
    pc.resolution_x = W;
    pc.resolution_y = H;
    pc.seed = 42;
    pc.time = 0.0f;

    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u) << "readback submit failed";

    std::vector<float> out;
    uint32_t ow = 0, oh = 0;
    uint64_t out_gen = 0;
    ASSERT_TRUE(wait_for_readback(engine, out, ow, oh, out_gen));
    EXPECT_EQ(ow, W);
    EXPECT_EQ(oh, H);
    EXPECT_EQ(out.size(), W * H * 4u);

    // The output MUST NOT be black. If chain_in_sampled_slots is stale
    // (pointing to the 1x1 dummy), the shader samples the dummy and
    // outputs black. With the fix, the shader reads the live slot from
    // pe.in_sampled_slots and outputs the real image.
    uint32_t cx = W / 2, cy = H / 2;
    uint32_t ci = (cy * W + cx) * 4;
    EXPECT_NEAR(out[ci + 0], 1.0f, 0.05f)
        << "R channel mismatch — chain_in_sampled_slots likely stale (dummy)";
    EXPECT_NEAR(out[ci + 1], 0.5f, 0.05f)
        << "G channel mismatch — chain_in_sampled_slots likely stale (dummy)";
    EXPECT_NEAR(out[ci + 2], 0.25f, 0.05f)
        << "B channel mismatch — chain_in_sampled_slots likely stale (dummy)";

    engine.shutdown();
}
