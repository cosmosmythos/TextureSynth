// Dedicated correctness tests for the color_const node.
// GLSL contract (shader_assets/nodes/color_const.glsl):
//   mode < 0.5  -> vec4(r, r, r, 1.0)   // Mono mode, alpha forced to 1
//   else        -> vec4(r, g, b, a)      // RGBA mode, alpha from param
// params order (color_const.node.json): [mode, r, g, b, a]
#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "test_assets.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <cmath>
#include <functional>

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

bool init_engine(Engine& engine, const char* cache_name) {
    return engine.init(VK_NULL_HANDLE, nullptr, 0, true, cache_name,
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
    pc.resolution_x = engine.output().extent().width;
    pc.resolution_y = engine.output().extent().height;
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

// Return center-pixel RGBA. Output is a constant field, so one pixel is enough.
void center_rgba(const std::vector<float>& px, uint32_t w, uint32_t h,
                 float& r, float& g, float& b, float& a) {
    size_t c = ((h / 2) * w + (w / 2)) * 4;
    r = px[c + 0]; g = px[c + 1]; b = px[c + 2]; a = px[c + 3];
}

// Tolerance accounts for F16 storage quantization.
void expect_uniform(const std::vector<float>& px,
                    float r, float g, float b, float a, float tol = 1e-2f) {
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        EXPECT_NEAR(px[i + 0], r, tol) << "R mismatch @ pixel " << (i / 4);
        EXPECT_NEAR(px[i + 1], g, tol) << "G mismatch @ pixel " << (i / 4);
        EXPECT_NEAR(px[i + 2], b, tol) << "B mismatch @ pixel " << (i / 4);
        EXPECT_NEAR(px[i + 3], a, tol) << "A mismatch @ pixel " << (i / 4);
        if (i == 0) break;  // one pixel enough; field is constant
    }
}

} // namespace

class ColorConst : public ::testing::Test {
protected:
    Engine engine;
    NodeLibrary lib = load_real_lib();
    bool initialized = false;

    void SetUp() override {
        if (!init_engine(engine, "test_color_const")) GTEST_SKIP() << engine.last_error();
        initialized = true;
    }

    void TearDown() override {
        if (initialized) { engine.shutdown(); initialized = false; }
    }

    // mode, r, g, b, a in JSON param order.
    uint64_t build_and_set(float mode, float r, float g, float b, float a) {
        Graph graph;
        graph.nodes.push_back({1, "color_const"});
        graph.output_node = 1;
        uint64_t gen = engine.set_graph(graph);
        if (gen == 0u) return 0;
        if (!wait_for_pipeline(engine)) return 0;
        engine.update_node_params_by_id(1, {mode, r, g, b, a});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        return gen;
    }
};

// ── RGBA mode: output = (r, g, b, a) exactly ──

TEST_F(ColorConst, RGBA_Red) {
    uint64_t gen = build_and_set(/*mode*/1.0f, 1.0f, 0.0f, 0.0f, 1.0f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 1.0f, 0.0f, 0.0f, 1.0f);
}

TEST_F(ColorConst, RGBA_Green) {
    uint64_t gen = build_and_set(1.0f, 0.0f, 1.0f, 0.0f, 1.0f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 0.0f, 1.0f, 0.0f, 1.0f);
}

TEST_F(ColorConst, RGBA_Blue) {
    uint64_t gen = build_and_set(1.0f, 0.0f, 0.0f, 1.0f, 1.0f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 0.0f, 0.0f, 1.0f, 1.0f);
}

TEST_F(ColorConst, RGBA_ArbitraryValues) {
    uint64_t gen = build_and_set(1.0f, 0.25f, 0.5f, 0.75f, 0.4f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 0.25f, 0.5f, 0.75f, 0.4f);
}

// HDR > 1.0 values must survive (Blend was rev'd to HDR-merge semantics).
TEST_F(ColorConst, RGBA_HDR_AboveOne) {
    uint64_t gen = build_and_set(1.0f, 2.0f, 4.5f, 0.0f, 1.0f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 2.0f, 4.5f, 0.0f, 1.0f);
}

// Alpha = 0 in RGBA mode -> (r,g,b,0). No premultiply.
TEST_F(ColorConst, RGBA_AlphaZero_NoPremultiply) {
    uint64_t gen = build_and_set(1.0f, 1.0f, 0.0f, 0.0f, 0.0f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 1.0f, 0.0f, 0.0f, 0.0f);
}

// ── Mono mode: output = (r, r, r, 1.0); alpha param ignored, forced to 1 ──

TEST_F(ColorConst, Mono_Gray) {
    uint64_t gen = build_and_set(/*mode*/0.0f, 0.5f, 0.0f, 0.0f, 0.0f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 0.5f, 0.5f, 0.5f, 1.0f);
}

TEST_F(ColorConst, Mono_AlphaParamIgnored) {
    // a=0.2 in params, but Mono mode forces A=1.0; g,b ignored, all = r.
    uint64_t gen = build_and_set(0.0f, 0.3f, 0.9f, 0.7f, 0.2f);
    ASSERT_NE(gen, 0u);
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 0.3f, 0.3f, 0.3f, 1.0f);
}

// ── Defaults: no update_node_params -> all 1.0 ──

TEST_F(ColorConst, Defaults_AllOne) {
    Graph g;
    g.nodes.push_back({1, "color_const"});
    g.output_node = 1;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    expect_uniform(px, 1.0f, 1.0f, 1.0f, 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Blend-integration tests: color_const → Blend A or B, various masks + formats.
// Root bug: mask value leaks into G/B of blend output when A or B source is
// connected. These tests reproduce the exact scenarios and assert
// that G/B stay correct (no mask leakage).
//
// Color_const SSBO layout: [mode, r, g, b, a]
//   mode=1 -> RGBA, mode=0 -> grayscale (r=r, g=r, b=r, a=1)
// Blend SSBO layout: [mode, mask]
//   mode=0 -> Mix
// ═══════════════════════════════════════════════════════════════════════════════

namespace {

struct BlendTestResult {
    float r, g, b, a;
    uint32_t w, h;
};

// Read back center pixel from an already-submitted graph generation.
// Does NOT call set_graph — caller owns the generation lifecycle.
BlendTestResult readback_center(Engine& engine, uint64_t gen,
                                const char* label) {
    BlendTestResult out{};
    std::vector<float> px;
    if (!render(engine, gen, px, out.w, out.h)) {
        std::cout << "[" << label << "] readback FAILED\n"; return out;
    }
    size_t c = ((out.h / 2) * out.w + (out.w / 2)) * 4;
    out.r = px[c]; out.g = px[c+1]; out.b = px[c+2]; out.a = px[c+3];
    std::cout << "[" << label << "] center RGBA = ("
              << out.r << ", " << out.g << ", " << out.b << ", " << out.a << ")\n";
    return out;
}

} // namespace

// ── T8: Color_const (RGBA red) → Blend A, B=empty, mask=slider ──
// mask=1.0 -> full A = (1,0,0,1). mask=0.0 -> full B = (0,0,0,1).
// mask=0.5 -> (0.5, 0, 0, 1). G and B must be 0 at all mask values.
TEST_F(ColorConst, BlendA_Red_SliderMask) {
    // mask=1.0 -> expect (1, 0, 0, 1)
    {
        Graph g;
        g.nodes.push_back({1, "color_const"});
        g.nodes.push_back({2, "blend"});
        g.connections.push_back({1, 0, 2, 1});  // color -> blend.a
        g.output_node = 2;
        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        ASSERT_TRUE(wait_for_pipeline(engine));
        engine.update_node_params_by_id(1, {1.0f, 1.0f, 0.0f, 0.0f, 1.0f});  // RGBA red
        engine.update_node_params_by_id(2, {0.0f, 1.0f});  // Mix, mask=1.0
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        auto r = readback_center(engine, gen, "T8.1 red, mask=1.0");
        EXPECT_NEAR(r.r, 1.0f, 1e-2f);
        EXPECT_NEAR(r.g, 0.0f, 1e-2f);  // G must be 0 — no mask leak
        EXPECT_NEAR(r.b, 0.0f, 1e-2f);  // B must be 0 — no mask leak
        EXPECT_NEAR(r.a, 1.0f, 1e-2f);
    }
    // mask=0.5 -> expect (0.5, 0, 0, 1)
    {
        Graph g;
        g.nodes.push_back({1, "color_const"});
        g.nodes.push_back({2, "blend"});
        g.connections.push_back({1, 0, 2, 1});
        g.output_node = 2;
        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        ASSERT_TRUE(wait_for_pipeline(engine));
        engine.update_node_params_by_id(1, {1.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        engine.update_node_params_by_id(2, {0.0f, 0.5f});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        auto r = readback_center(engine, gen, "T8.2 red, mask=0.5");
        EXPECT_NEAR(r.r, 0.5f, 1e-2f);
        EXPECT_NEAR(r.g, 0.0f, 1e-2f);  // G must be 0 — mask=0.5 must NOT leak
        EXPECT_NEAR(r.b, 0.0f, 1e-2f);  // B must be 0 — mask=0.5 must NOT leak
        EXPECT_NEAR(r.a, 1.0f, 1e-2f);
    }
    // mask=0.0 -> expect (0, 0, 0, 1)
    {
        Graph g;
        g.nodes.push_back({1, "color_const"});
        g.nodes.push_back({2, "blend"});
        g.connections.push_back({1, 0, 2, 1});
        g.output_node = 2;
        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        ASSERT_TRUE(wait_for_pipeline(engine));
        engine.update_node_params_by_id(1, {1.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        engine.update_node_params_by_id(2, {0.0f, 0.0f});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        auto r = readback_center(engine, gen, "T8.3 red, mask=0.0");
        EXPECT_NEAR(r.r, 0.0f, 1e-2f);
        EXPECT_NEAR(r.g, 0.0f, 1e-2f);
        EXPECT_NEAR(r.b, 0.0f, 1e-2f);
        EXPECT_NEAR(r.a, 1.0f, 1e-2f);
    }
}

// ── T9: Color_const (RGBA blue) → Blend A, B=empty, mask=noise (worley) ──
// A = blue (0,0,1,1). B = empty (0,0,0,1). Mask = worley noise.
// Blend Mix: output = mix((0,0,0), (0,0,1), f) = (0, 0, f, 1).
// G must be 0 everywhere. B varies with mask. A must be 1.
TEST_F(ColorConst, BlendA_Blue_WorleyMask) {
    Graph g;
    g.nodes.push_back({1, "color_const"});
    g.nodes.push_back({2, "worley"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});  // color -> blend.a
    g.connections.push_back({2, 0, 3, 0});  // worley -> blend.mask
    g.output_node = 3;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));
    engine.update_node_params_by_id(1, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f});  // RGBA blue
    engine.update_node_params_by_id(3, {0.0f});  // Mix, mask from worley
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(render(engine, gen, px, w, h));
    // Check ALL pixels: G must be 0 (no mask leak), A must be 1.
    float gmin = 1e9f, gmax = -1e9f;
    float amin = 1e9f, amax = -1e9f;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        gmin = std::min(gmin, px[i+1]); gmax = std::max(gmax, px[i+1]);
        amin = std::min(amin, px[i+3]); amax = std::max(amax, px[i+3]);
    }
    std::cout << "T9: G[" << gmin << ".." << gmax << "] A[" << amin << ".." << amax << "]\n";
    EXPECT_NEAR(gmin, 0.0f, 1e-2f);  // G must be 0 — no mask leak
    EXPECT_NEAR(gmax, 0.0f, 1e-2f);
    EXPECT_NEAR(amin, 1.0f, 1e-2f);  // A must be 1
    EXPECT_NEAR(amax, 1.0f, 1e-2f);
}

// ── T10: Color_const → Blend B (not A), mask=slider ──
// A = empty (0,0,0,1). B = red (1,0,0,1). mask=1.0 -> full A = (0,0,0,1).
// mask=0.0 -> full B = (1,0,0,1). mask=0.5 -> (0.5, 0, 0, 1).
TEST_F(ColorConst, BlendB_Red_SliderMask) {
    // mask=0.0 -> expect (1, 0, 0, 1) = full B
    {
        Graph g;
        g.nodes.push_back({1, "color_const"});
        g.nodes.push_back({2, "blend"});
        g.connections.push_back({1, 0, 2, 2});  // color -> blend.b (socket 2)
        g.output_node = 2;
        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        ASSERT_TRUE(wait_for_pipeline(engine));
        engine.update_node_params_by_id(1, {1.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        engine.update_node_params_by_id(2, {0.0f, 0.0f});  // Mix, mask=0.0 -> full B
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        auto r = readback_center(engine, gen, "T10.1 red in B, mask=0.0");
        EXPECT_NEAR(r.r, 1.0f, 1e-2f);
        EXPECT_NEAR(r.g, 0.0f, 1e-2f);
        EXPECT_NEAR(r.b, 0.0f, 1e-2f);
        EXPECT_NEAR(r.a, 1.0f, 1e-2f);
    }
    // mask=0.5 -> expect (0.5, 0, 0, 1)
    {
        Graph g;
        g.nodes.push_back({1, "color_const"});
        g.nodes.push_back({2, "blend"});
        g.connections.push_back({1, 0, 2, 2});
        g.output_node = 2;
        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        ASSERT_TRUE(wait_for_pipeline(engine));
        engine.update_node_params_by_id(1, {1.0f, 1.0f, 0.0f, 0.0f, 1.0f});
        engine.update_node_params_by_id(2, {0.0f, 0.5f});
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        auto r = readback_center(engine, gen, "T10.2 red in B, mask=0.5");
        EXPECT_NEAR(r.r, 0.5f, 1e-2f);
        EXPECT_NEAR(r.g, 0.0f, 1e-2f);  // no mask leak
        EXPECT_NEAR(r.b, 0.0f, 1e-2f);
        EXPECT_NEAR(r.a, 1.0f, 1e-2f);
    }
}

// ── T11: Color_const (RGBA) → Blend A, Color_const (RGBA) → Blend B, mask=slider ──
// A = red (1,0,0,1). B = green (0,1,0,1). mask=0.5 -> (0.5, 0.5, 0, 1).
TEST_F(ColorConst, BlendAB_RedGreen_SliderMask) {
    Graph g;
    g.nodes.push_back({1, "color_const"});  // A = red
    g.nodes.push_back({2, "color_const"});  // B = green
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});  // red -> blend.a
    g.connections.push_back({2, 0, 3, 2});  // green -> blend.b
    g.output_node = 3;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));
    engine.update_node_params_by_id(1, {1.0f, 1.0f, 0.0f, 0.0f, 1.0f});  // RGBA red
    engine.update_node_params_by_id(2, {1.0f, 0.0f, 1.0f, 0.0f, 1.0f});  // RGBA green
    engine.update_node_params_by_id(3, {0.0f, 0.5f});  // Mix, mask=0.5
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    auto r = readback_center(engine, gen, "T11 red+green, mask=0.5");
    EXPECT_NEAR(r.r, 0.5f, 1e-2f);
    EXPECT_NEAR(r.g, 0.5f, 1e-2f);
    EXPECT_NEAR(r.b, 0.0f, 1e-2f);
    EXPECT_NEAR(r.a, 1.0f, 1e-2f);
}
