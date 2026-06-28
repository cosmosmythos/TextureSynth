// Regression: Blend.001 with both A and B connected.
// When both inputs are connected, mask modulation breaks — output is constant.
// This test replicates the exact graph from the Blender scene and verifies
// that mask=0 outputs B, mask=1 outputs A, mask=0.5 outputs a mix.
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

using namespace te;

namespace {

NodeLibrary load_real_lib() {
    NodeLibrary lib;
    std::string err;
    int n = NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    EXPECT_GT(n, 0) << "failed to load real nodes: " << err;
    return lib;
}

bool init_engine(Engine& engine) {
    return engine.init(VK_NULL_HANDLE, nullptr, 0, true, "cache_blend_dual",
                       find_test_nodes_dir().c_str(),
                       find_test_glsl_dir().c_str());
}

bool wait_for_pipeline(Engine& engine, int timeout_ms = 5000) {
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        engine.poll_pending_compiles();
        if (engine.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return engine.has_pipeline();
}

bool render(Engine& engine, uint64_t gen,
            std::vector<float>& px, uint32_t& w, uint32_t& h,
            int timeout_ms = 5000) {
    if (!wait_for_pipeline(engine)) return false;
    PushConstants pc{};
    pc.resolution_x = 64; pc.resolution_y = 64;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    if (ticket == 0) return false;
    uint64_t og = 0;
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        if (engine.async_readback().poll(engine.ctx(), px, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

float mean_r(const std::vector<float>& px) {
    double sum = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) sum += px[i];
    return (float)(sum / (px.size() / 4));
}

// Build the exact graph from the Blender scene.
// Node IDs map to stable_id() values, but any unique IDs work.
//
// Graph topology (matches Blender exactly):
//   Worley(1) -> Levels(2) -> Blur(3) -> Invert(4) -> Blend(9).A
//   Levels(2) -> Invert.001(5) -> Blur.001(6) -> Blend.001(12).A
//   Perlin(7) -> Levels.001(8) -> Blend(9).B
//   Blend(9) -> Levels.002(10) -> Blur.002(11) -> Blend.001(12).B
//   Output = Blend.001(12)
Graph build_full_graph() {
    Graph g;
    g.nodes.push_back({1,  "worley"});
    g.nodes.push_back({2,  "levels"});
    g.nodes.push_back({3,  "blur"});
    g.nodes.push_back({4,  "invert"});
    g.nodes.push_back({5,  "invert"});
    g.nodes.push_back({6,  "blur"});
    g.nodes.push_back({7,  "perlin"});
    g.nodes.push_back({8,  "levels"});
    g.nodes.push_back({9,  "blend"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blur"});
    g.nodes.push_back({12, "blend"});

    // Connections (from Blender introspection):
    g.connections.push_back({1, 0, 2, 0});   // Worley -> Levels.in[0]
    g.connections.push_back({2, 0, 3, 0});   // Levels -> Blur.in[0]
    g.connections.push_back({2, 0, 5, 1});   // Levels -> Invert.001.in[1] (Color)
    g.connections.push_back({3, 0, 4, 1});   // Blur -> Invert.in[1] (Color)
    g.connections.push_back({4, 0, 9, 1});   // Invert -> Blend.in[1] (A)
    g.connections.push_back({7, 0, 8, 0});   // Perlin -> Levels.001.in[0]
    g.connections.push_back({8, 0, 9, 2});   // Levels.001 -> Blend.in[2] (B)
    g.connections.push_back({9, 0, 10, 0});  // Blend -> Levels.002.in[0]
    g.connections.push_back({10, 0, 11, 0}); // Levels.002 -> Blur.002.in[0]
    g.connections.push_back({5, 0, 6, 0});   // Invert.001 -> Blur.001.in[0]
    g.connections.push_back({6, 0, 12, 1});  // Blur.001 -> Blend.001.in[1] (A)
    g.connections.push_back({11, 0, 12, 2}); // Blur.002 -> Blend.001.in[2] (B)

    g.output_node = 12;
    return g;
}

void set_all_params(Engine& engine) {
    // Exact params from Blender introspection:
    engine.update_node_params_by_id(1, {8.0f, 3.0f, 2.0f, 0.5f, 1.0f, 0.0f, 81.0f});
    engine.update_node_params_by_id(7, {31.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f});
    engine.update_node_params_by_id(3, {0.1f});
    engine.update_node_params_by_id(6, {0.05f});
    engine.update_node_params_by_id(11, {0.03f});
    engine.update_node_params_by_id(4, {1.0f});
    engine.update_node_params_by_id(5, {1.0f});
    engine.update_node_params_by_id(9, {1.0f, 0.4533f});
    engine.update_node_params_by_id(12, {0.0f, 0.5f});
    // Levels params (26 floats each, all R/G/B/A channel curves):
    std::vector<float> levels_p = {0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f,
        0.42f, 0.61f, 1.0f, 0.0f, 0.27f, 0.5f, 0.61f, 0.0f, 1.0f,
        0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(2, levels_p);
    std::vector<float> levels001_p = {0.42f, 0.3267f, 0.34f, 0.0f, 0.31f,
        0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.42f, 0.3267f, 0.34f, 0.0f,
        0.31f, 0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(8, levels001_p);
    std::vector<float> levels002_p = {0.162f, 0.5067f, 0.76f, 0.0f, 0.49f,
        0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.162f, 0.5067f, 0.76f, 0.0f,
        0.49f, 0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(10, levels002_p);
}

} // namespace


// Test 1: Full graph, mask=0 — output should equal B (Blur.001).
TEST(BlendDualInput, FullGraph_Mask0_OutputEqualsB) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    Graph g = build_full_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    set_all_params(engine);
    // Set mask=0: [mode=0(MIX), mask=0.0]
    engine.update_node_params_by_id(12, {0.0f, 0.0f});

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    ASSERT_FALSE(px.empty());

    float r = mean_r(px);
    std::cout << "FullGraph mask=0: R mean = " << r << std::endl;

    // Now render B alone (Blur.002) for reference.
    // We need Blur.002's output, so set active node to node 11.
    uint64_t gen_b = engine.set_active_node(11);
    ASSERT_NE(gen_b, 0u) << engine.last_error();

    std::vector<float> px_b;
    ASSERT_TRUE(render(engine, gen_b, px_b, w, h));
    float r_b = mean_r(px_b);
    std::cout << "Blur.002 alone: R mean = " << r_b << std::endl;

    // mask=0 with MIX mode: output = B
    EXPECT_NEAR(r, r_b, 0.05f)
        << "Blend.001 mask=0 should output B (Blur.001), but got " << r
        << " vs B=" << r_b;
}


// Test 2: Full graph, mask=1 — output should equal A (Blur.002).
TEST(BlendDualInput, FullGraph_Mask1_OutputEqualsA) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    Graph g = build_full_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    set_all_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 1.0f});

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    ASSERT_FALSE(px.empty());

    float r = mean_r(px);
    std::cout << "FullGraph mask=1: R mean = " << r << std::endl;

    // Render A alone (Blur.001) for reference.
    uint64_t gen_a = engine.set_active_node(6);
    ASSERT_NE(gen_a, 0u) << engine.last_error();

    std::vector<float> px_a;
    ASSERT_TRUE(render(engine, gen_a, px_a, w, h));
    float r_a = mean_r(px_a);
    std::cout << "Blur.001 alone: R mean = " << r_a << std::endl;

    // mask=1 with MIX mode: output = A
    EXPECT_NEAR(r, r_a, 0.05f)
        << "Blend.001 mask=1 should output A (Blur.002), but got " << r
        << " vs A=" << r_a;
}


// Test 3: Verify mask modulation — R should change between mask=0 and mask=1.
TEST(BlendDualInput, FullGraph_MaskModulation) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    Graph g = build_full_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    set_all_params(engine);

    // mask=0
    engine.update_node_params_by_id(12, {0.0f, 0.0f});
    std::vector<float> px0;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(render(engine, gen, px0, w, h));
    float r0 = mean_r(px0);

    // mask=1
    engine.update_node_params_by_id(12, {0.0f, 1.0f});
    std::vector<float> px1;
    ASSERT_TRUE(render(engine, gen, px1, w, h));
    float r1 = mean_r(px1);

    // mask=0.5
    engine.update_node_params_by_id(12, {0.0f, 0.5f});
    std::vector<float> px05;
    ASSERT_TRUE(render(engine, gen, px05, w, h));
    float r05 = mean_r(px05);

    std::cout << "Mask modulation: mask=0 -> R=" << r0
              << ", mask=0.5 -> R=" << r05
              << ", mask=1 -> R=" << r1 << std::endl;

    // R should differ between mask=0 and mask=1 (unless A==B which is unlikely).
    EXPECT_NE(r0, r1) << "mask=0 and mask=1 produced identical output — mask not modulating!";
    // mask=0.5 should be between mask=0 and mask=1 (for MIX mode).
    float lo = std::min(r0, r1);
    float hi = std::max(r0, r1);
    EXPECT_GE(r05, lo - 0.02f) << "mask=0.5 below both extremes";
    EXPECT_LE(r05, hi + 0.02f) << "mask=0.5 above both extremes";
}


// Test 4: Replicate addon readback_sync path exactly.
// The addon calls update_node_params_by_id then readback_sync.
// readback_sync creates its own PushConstants and calls async_.drain + async_.submit.
// This tests whether that path preserves param changes.
TEST(BlendDualInput, ReadbackSync_MaskModulation) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    Graph g = build_full_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    set_all_params(engine);

    // mask=0
    engine.update_node_params_by_id(12, {0.0f, 0.0f});
    auto img0 = engine.readback_sync();
    ASSERT_FALSE(img0.pixels.empty());
    float r0 = 0;
    for (size_t i = 0; i + 3 < img0.pixels.size(); i += 4) r0 += img0.pixels[i];
    r0 /= (img0.pixels.size() / 4);

    // mask=1
    engine.update_node_params_by_id(12, {0.0f, 1.0f});
    auto img1 = engine.readback_sync();
    ASSERT_FALSE(img1.pixels.empty());
    float r1 = 0;
    for (size_t i = 0; i + 3 < img1.pixels.size(); i += 4) r1 += img1.pixels[i];
    r1 /= (img1.pixels.size() / 4);

    // mask=0.5
    engine.update_node_params_by_id(12, {0.0f, 0.5f});
    auto img05 = engine.readback_sync();
    ASSERT_FALSE(img05.pixels.empty());
    float r05 = 0;
    for (size_t i = 0; i + 3 < img05.pixels.size(); i += 4) r05 += img05.pixels[i];
    r05 /= (img05.pixels.size() / 4);

    std::cout << "ReadbackSync: mask=0 -> R=" << r0
              << ", mask=0.5 -> R=" << r05
              << ", mask=1 -> R=" << r1 << std::endl;

    EXPECT_NE(r0, r1) << "readback_sync: mask=0 and mask=1 identical — param not reaching shader!";
    float lo = std::min(r0, r1);
    float hi = std::max(r0, r1);
    EXPECT_GE(r05, lo - 0.02f) << "readback_sync: mask=0.5 below both extremes";
    EXPECT_LE(r05, hi + 0.02f) << "readback_sync: mask=0.5 above both extremes";
}


// Test 5: Regression — adding an UNRELATED link corrupts Blend.001 output.
//
// Scenario from Blender (DEV_LOG/blend001_investigation.md):
//   1. Build graph with Blend.001 B-only (mask=0) → output = Blur.002 data
//   2. Add Blur.001 → Blend.002.A (completely unrelated node)
//   3. Rebuild graph via set_graph()
//   4. Blend.001 output changes from Blur.002 data to Blur.001 data
//
// Root cause hypothesis: set_graph() reassigns Blend.001's input resource
// to the wrong VkImage after topology change.
Graph build_b_only_graph() {
    Graph g;
    // Same 12 nodes as the full graph
    g.nodes.push_back({1,  "worley"});
    g.nodes.push_back({2,  "levels"});
    g.nodes.push_back({3,  "blur"});
    g.nodes.push_back({4,  "invert"});
    g.nodes.push_back({5,  "invert"});
    g.nodes.push_back({6,  "blur"});
    g.nodes.push_back({7,  "perlin"});
    g.nodes.push_back({8,  "levels"});
    g.nodes.push_back({9,  "blend"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blur"});
    g.nodes.push_back({12, "blend"});

    // Connections — same as full graph EXCEPT:
    //   - Blend.001.A is NOT connected (no Blur.001→Blend.001.A)
    //   - Blend.002 exists as node 13 with mask=1, no inputs connected
    g.connections.push_back({1, 0, 2, 0});   // Worley -> Levels.in[0]
    g.connections.push_back({2, 0, 3, 0});   // Levels -> Blur.in[0]
    g.connections.push_back({2, 0, 5, 1});   // Levels -> Invert.001.in[1] (Color)
    g.connections.push_back({3, 0, 4, 1});   // Blur -> Invert.in[1] (Color)
    g.connections.push_back({4, 0, 9, 1});   // Invert -> Blend.in[1] (A)
    g.connections.push_back({7, 0, 8, 0});   // Perlin -> Levels.001.in[0]
    g.connections.push_back({8, 0, 9, 2});   // Levels.001 -> Blend.in[2] (B)
    g.connections.push_back({9, 0, 10, 0});  // Blend -> Levels.002.in[0]
    g.connections.push_back({10, 0, 11, 0}); // Levels.002 -> Blur.002.in[0]
    g.connections.push_back({5, 0, 6, 0});   // Invert.001 -> Blur.001.in[0]
    // NOTE: Blur.001(6) -> Blend.001(12).A is NOT connected
    g.connections.push_back({11, 0, 12, 2}); // Blur.002 -> Blend.001.in[2] (B)

    g.output_node = 12;
    return g;
}

Graph build_b_only_graph_with_unrelated_link() {
    Graph g;
    // 13 nodes — add Blend.002 as node 13
    g.nodes.push_back({1,  "worley"});
    g.nodes.push_back({2,  "levels"});
    g.nodes.push_back({3,  "blur"});
    g.nodes.push_back({4,  "invert"});
    g.nodes.push_back({5,  "invert"});
    g.nodes.push_back({6,  "blur"});
    g.nodes.push_back({7,  "perlin"});
    g.nodes.push_back({8,  "levels"});
    g.nodes.push_back({9,  "blend"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blur"});
    g.nodes.push_back({12, "blend"});
    g.nodes.push_back({13, "blend"});  // Blend.002 — the "unrelated" node

    // Same connections as B-only, PLUS Blur.001 -> Blend.002.A
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
    g.connections.push_back({11, 0, 12, 2});
    // THE UNRELATED LINK — Blur.001 -> Blend.002.A (node 13, input 1)
    g.connections.push_back({6, 0, 13, 1});
    // Blend.002 mask=1, B not connected

    g.output_node = 12;
    return g;
}

TEST(BlendDualInput, UnrelatedLink_TopologyChange_CorruptsBlend001) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    // Step 1: Build B-only graph (no A connected to Blend.001)
    Graph g1 = build_b_only_graph();
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_NE(gen1, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);
    // Blend.001: mask=0 → output should be B (Blur.002 data)
    engine.update_node_params_by_id(12, {0.0f, 0.0f});

    auto img1 = engine.readback_sync();
    ASSERT_FALSE(img1.pixels.empty());
    float r1 = 0;
    for (size_t i = 0; i + 3 < img1.pixels.size(); i += 4) r1 += img1.pixels[i];
    r1 /= (img1.pixels.size() / 4);

    // Also get Blur.002 reference
    engine.set_active_node(11);
    auto img_b = engine.readback_sync();
    float r_b = 0;
    for (size_t i = 0; i + 3 < img_b.pixels.size(); i += 4) r_b += img_b.pixels[i];
    r_b /= (img_b.pixels.size() / 4);

    // Reset active to output node
    engine.set_active_node(12);

    std::cout << "Step 1 (B-only): Blend.001 R=" << r1
              << ", Blur.002 R=" << r_b << std::endl;
    EXPECT_NEAR(r1, r_b, 0.01f)
        << "B-only: Blend.001 should output Blur.002 data";

    // Step 2: Rebuild with UNRELATED link added
    Graph g2 = build_b_only_graph_with_unrelated_link();
    uint64_t gen2 = engine.set_graph(g2);
    ASSERT_NE(gen2, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Same params (set_graph may reset param SSBO)
    set_all_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 0.0f});

    auto img2 = engine.readback_sync();
    ASSERT_FALSE(img2.pixels.empty());
    float r2 = 0;
    for (size_t i = 0; i + 3 < img2.pixels.size(); i += 4) r2 += img2.pixels[i];
    r2 /= (img2.pixels.size() / 4);

    std::cout << "Step 2 (after unrelated link): Blend.001 R=" << r2 << std::endl;
    std::cout << "  Step 1 was R=" << r1 << " (correct Blur.002 data)" << std::endl;

    // CRITICAL: Blend.001 output must NOT change just because an unrelated
    // node's link was added. Same params, same B-only input → same output.
    EXPECT_NEAR(r2, r1, 0.02f)
        << "BUG: Adding Blur.001->Blend.002.A (unrelated) changed Blend.001 output!"
        << " Expected R=" << r1 << " got R=" << r2;
}


// Test 6: Full-graph corruption — add Blend.002.A link to the working full graph.
// This is the exact Blender scenario: graph works, then adding an unrelated
// link to Blend.002 changes Blend.001's output.
Graph build_full_graph_with_blend002() {
    Graph g = build_full_graph();
    // Add Blend.002 as node 13 with Blur.001 -> Blend.002.A
    g.nodes.push_back({13, "blend"});
    g.connections.push_back({6, 0, 13, 1});  // Blur.001 -> Blend.002.A
    return g;
}

TEST(BlendDualInput, FullGraph_AddBlend002Link_ChangesBlend001) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    // Step 1: Build full graph (both A and B connected, no Blend.002)
    Graph g1 = build_full_graph();
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_NE(gen1, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);
    // Blend.001: mask=0 → output = B
    engine.update_node_params_by_id(12, {0.0f, 0.0f});

    auto img1 = engine.readback_sync();
    float r1 = 0;
    for (size_t i = 0; i + 3 < img1.pixels.size(); i += 4) r1 += img1.pixels[i];
    r1 /= (img1.pixels.size() / 4);

    // Get Blur.002 reference (node 11)
    engine.set_active_node(11);
    auto img_ref = engine.readback_sync();
    float r_ref = 0;
    for (size_t i = 0; i + 3 < img_ref.pixels.size(); i += 4) r_ref += img_ref.pixels[i];
    r_ref /= (img_ref.pixels.size() / 4);
    engine.set_active_node(12);

    std::cout << "FullGraph step 1: Blend.001 R=" << r1
              << ", Blur.002 R=" << r_ref << std::endl;

    // Step 2: Rebuild with Blend.002 added (Blur.001 -> Blend.002.A)
    Graph g2 = build_full_graph_with_blend002();
    uint64_t gen2 = engine.set_graph(g2);
    ASSERT_NE(gen2, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 0.0f});

    auto img2 = engine.readback_sync();
    float r2 = 0;
    for (size_t i = 0; i + 3 < img2.pixels.size(); i += 4) r2 += img2.pixels[i];
    r2 /= (img2.pixels.size() / 4);

    std::cout << "FullGraph step 2 (after adding Blend.002): Blend.001 R=" << r2 << std::endl;

    // Blend.001 output must NOT change — same inputs, same params
    EXPECT_NEAR(r2, r1, 0.02f)
        << "BUG: Adding Blend.002.A link changed Blend.001 output from R="
        << r1 << " to R=" << r2;
}
