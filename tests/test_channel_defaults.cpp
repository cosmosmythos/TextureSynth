// Reproduction tests for channel-format readback + unconnected-socket defaults.
// Each test prints actual RGBA values so the bug is visible in the test log.
// Expected (user rule):
//   Mono read in vec4 context -> (R, 0, 0, 1)
//   Empty color socket        -> (0, 0, 0, 1)
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

bool wait_for_readback_gen(Engine& engine, uint64_t gen,
                           std::vector<float>& pixels,
                           uint32_t& w, uint32_t& h, int timeout_ms = 5000) {
    PushConstants pc{};
    pc.resolution_x = 32; pc.resolution_y = 32;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    if (ticket == 0) return false;
    uint64_t og = 0;
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

// Print RGBA at the center pixel + per-channel min/max across the image.
void dump_rgba(const std::vector<float>& px, uint32_t w, uint32_t h, const char* label) {
    if (px.empty()) { std::cout << "[" << label << "] empty\n"; return; }
    std::cout << "[" << label << "] dims=" << w << "x" << h
              << " pixels=" << (px.size() / 4) << "\n";
    size_t center = ((h / 2) * w + (w / 2)) * 4;
    float rmin = 1e9, rmax = -1e9, gmin = 1e9, gmax = -1e9, bmin = 1e9, bmax = -1e9, amin = 1e9, amax = -1e9;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        rmin = std::min(rmin, px[i]); rmax = std::max(rmax, px[i]);
        gmin = std::min(gmin, px[i+1]); gmax = std::max(gmax, px[i+1]);
        bmin = std::min(bmin, px[i+2]); bmax = std::max(bmax, px[i+2]);
        amin = std::min(amin, px[i+3]); amax = std::max(amax, px[i+3]);
    }
    std::cout << "[" << label << "] center RGBA = ("
              << px[center] << ", " << px[center+1] << ", "
              << px[center+2] << ", " << px[center+3] << ")\n"
              << "           R[" << rmin << ".." << rmax << "]  "
              << "G[" << gmin << ".." << gmax << "]  "
              << "B[" << bmin << ".." << bmax << "]  "
              << "A[" << amin << ".." << amax << "]\n";
}
} // namespace

class ChannelDefaults : public ::testing::Test {
protected:
    Engine engine;
    NodeLibrary lib = load_real_lib();
    void SetUp() override {
        if (!init_engine(engine, "test_chan_defaults")) GTEST_SKIP() << engine.last_error();
    }
};

// T1: value noise with format_override=Mono -> readback.
// Rule: should read as (R, 0, 0, 1).
TEST_F(ChannelDefaults, T1_MonoNoise_Readback) {
    Graph g;
    g.nodes.push_back({1, "value", ChannelFormat::Mono});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    dump_rgba(px, w, h, "T1 Mono noise");

    // Print-only for now; once we agree on values we add EXPECT lines.
    SUCCEED();
}

// T2: value noise with format_override=UV -> readback.
// Rule: should read as (R, G, 0, 1).
TEST_F(ChannelDefaults, T2_UVNoise_Readback) {
    Graph g;
    g.nodes.push_back({1, "value", ChannelFormat::UV});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    dump_rgba(px, w, h, "T2 UV noise");

    SUCCEED();
}

// T3: Blend with A and B unconnected, mask=1.0, mode=Mix (0).
// Rule: empty color socket should read as (0,0,0,1).
TEST_F(ChannelDefaults, T3_BlendEmptyAB_Readback) {
    Graph g;
    g.nodes.push_back({1, "blend"});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Blend SSBO layout = [mode, mask]. mode=0 (Mix), mask=1.0 (full A).
    engine.update_node_params_by_id(1, {0.0f, 1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    dump_rgba(px, w, h, "T3 Blend empty A/B");

    // Empty Blend a/b sockets must read opaque black (0,0,0,1), not transparent (0,0,0,0).
    size_t center = ((h / 2) * w + (w / 2)) * 4;
    EXPECT_NEAR(px[center + 0], 0.0f, 1e-5f);
    EXPECT_NEAR(px[center + 1], 0.0f, 1e-5f);
    EXPECT_NEAR(px[center + 2], 0.0f, 1e-5f);
    EXPECT_NEAR(px[center + 3], 1.0f, 1e-5f);
}

// T4: Bug #2 exact graph per user report: BOTH A and B are Mono, mask=Worley, mode=Multiply.
// If Mono reads correctly as (R,0,0,1) inside blend, then G and B MUST be 0 everywhere
// (Mono*Mono has no G/B; mix of two Mono has no G/B). Non-zero G/B = the leak.
TEST_F(ChannelDefaults, T4_MonoAB_WorleyMask_Readback) {
    Graph g;
    g.nodes.push_back({1, "value", ChannelFormat::Mono});   // A source (Mono)
    g.nodes.push_back({2, "value", ChannelFormat::Mono});   // B source (Mono)
    g.nodes.push_back({3, "worley"});                       // mask source (RGBA)
    g.nodes.push_back({4, "blend"});
    g.connections.push_back({1, 0, 4, 1});  // value -> blend.a
    g.connections.push_back({2, 0, 4, 2});  // value -> blend.b
    g.connections.push_back({3, 0, 4, 0});  // worley -> blend.mask
    g.output_node = 4;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Give B a different seed so A and B are distinguishable.
    engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
    // Blend SSBO layout = [mode, mask]. Mask connected (worley), only mode sent.
    // mode=3 (Multiply).
    engine.update_node_params_by_id(4, {3.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px; uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    dump_rgba(px, w, h, "T4 MonoA/MonoB/WorleyMask, Multiply");

    SUCCEED();
}

// T5: Reproduces the user's exact Bug #1 sequence on the Mono graph:
//   step 1: A=Mono, B=Mono, mask=slider (SSBO). Slider at 1.0 -> full A.
//   step 2: connect Worley -> mask. Mask now comes from texture.
//   step 3: disconnect mask -> mask=slider again. Slider at 0.0 then 1.0.
// Each step is its own set_graph (topology changes between them) and its own
// readback. We print center RGBA + ranges at every step. The questions:
//   Q1 (Bug #1): does step 3's slider still affect output?
//   Q2 (Bug #2): does any step show non-zero G/B from Mono sources?
TEST_F(ChannelDefaults, T5_MonoAB_MaskSliderConnectDisconnect) {
    auto render_step = [&](const char* label, Graph& g,
                           std::function<void()> after_params) -> bool {
        uint64_t gen = engine.set_graph(g);
        if (gen == 0u) { std::cout << "[" << label << "] set_graph FAILED\n"; return false; }
        if (!wait_for_pipeline(engine)) { std::cout << "[" << label << "] no pipeline\n"; return false; }
        after_params();
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        engine.poll_pending_compiles();
        std::vector<float> px; uint32_t w=0, h=0;
        if (!wait_for_readback_gen(engine, gen, px, w, h)) {
            std::cout << "[" << label << "] readback FAILED\n"; return false;
        }
        dump_rgba(px, w, h, label);
        return true;
    };

    // Two distinct Mono value sources.
    auto make_base = [](Graph& g) {
        g.nodes.push_back({1, "value", ChannelFormat::Mono});   // A
        g.nodes.push_back({2, "value", ChannelFormat::Mono});   // B
        g.nodes.push_back({3, "blend"});
        g.connections.push_back({1, 0, 3, 1});  // value -> blend.a
        g.connections.push_back({2, 0, 3, 2});  // value -> blend.b
        g.output_node = 3;
    };

    // ── Step 1: mask = slider, mode=Mix, mask=1.0 (expect full A) ──
    {
        Graph g; make_base(g);
        ASSERT_TRUE(render_step("T5.1 slider mask=1.0 (expect full A)", g, [&]{
            engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});  // B seed
            engine.update_node_params_by_id(3, {0.0f, 1.0f});  // mode=Mix, mask=1.0
        }));
    }
    // ── Step 1b: same graph, slider mask=0.0 (expect full B = seed-99 noise) ──
    {
        Graph g; make_base(g);
        ASSERT_TRUE(render_step("T5.1b slider mask=0.0 (expect full B)", g, [&]{
            engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
            engine.update_node_params_by_id(3, {0.0f, 0.0f});  // mode=Mix, mask=0.0
        }));
    }

    // ── Step 2: connect Worley -> mask. Mask is now the texture. ──
    {
        Graph g; make_base(g);
        g.nodes.push_back({4, "worley"});
        g.connections.push_back({4, 0, 3, 0});  // worley -> blend.mask
        g.output_node = 3;
        ASSERT_TRUE(render_step("T5.2 worley->mask connected", g, [&]{
            engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
            engine.update_node_params_by_id(3, {0.0f});  // mode only; mask comes from worley
        }));
    }

    // ── Step 3: disconnect mask -> back to slider, mask=1.0 (expect full A again) ──
    {
        Graph g; make_base(g);  // no worley connection — identical to step 1
        ASSERT_TRUE(render_step("T5.3 slider mask=1.0 AFTER disconnect (expect full A)", g, [&]{
            engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
            engine.update_node_params_by_id(3, {0.0f, 1.0f});
        }));
    }
    // ── Step 3b: slider mask=0.0 AFTER disconnect (expect full B again) ──
    {
        Graph g; make_base(g);
        ASSERT_TRUE(render_step("T5.3b slider mask=0.0 AFTER disconnect (expect full B)", g, [&]{
            engine.update_node_params_by_id(2, {8.0f, 5.0f, 2.0f, 0.5f, 0.0f, 99.0f});
            engine.update_node_params_by_id(3, {0.0f, 0.0f});
        }));
    }

    SUCCEED();
}
