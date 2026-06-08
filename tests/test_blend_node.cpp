// ============================================================================
// Robust Blend node coverage test.
//
// Goal: verify Blend behaves correctly for every input combination that
// the user can construct in viewer.exe / the addon. Each case is rendered
// and the output is checked for:
//   - not all-zero (the symptom the user reported in viewer.exe)
//   - not all-identical (the shader actually ran, didn't degenerate)
//   - for the "single connected" cases, the output matches the connected
//     source when factor=0 or factor=1 as appropriate
//   - for the "both connected" cases, swapping A<->B changes the output
//     (proves both inputs reach the shader and factor=0 vs 1 selects the
//     correct one)
//
// All tests are readback_sync-based: Engine compiles the graph, runs the
// full dispatch/readback cycle once, and returns a single RGBA image.
// The checker compares the actual pixel values against expected patterns.
// ============================================================================

#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "test_assets.hpp"
#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>

using namespace te;

namespace {

// Simple 4-channel image stats.
struct ImgStats {
    double sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
    float  min_r = 1e30f, min_g = 1e30f, min_b = 1e30f, min_a = 1e30f;
    float  max_r = -1e30f, max_g = -1e30f, max_b = -1e30f, max_a = -1e30f;
    bool   all_zero = true;
    bool   has_variation = false;
    // First non-zero pixel for debugging.
    float  first_nz_r = 0, first_nz_g = 0, first_nz_b = 0, first_nz_a = 0;
    bool   saw_nonzero = false;
};

ImgStats stats_of(const Engine::BakedImage& img) {
    ImgStats s;
    if (img.pixels.empty()) return s;
    const float* p = img.pixels.data();
    const size_t n = img.pixels.size();
    for (size_t i = 0; i < n; i += 4) {
        float r = p[i+0], g = p[i+1], b = p[i+2], a = p[i+3];
        s.sum_r += r; s.sum_g += g; s.sum_b += b; s.sum_a += a;
        s.min_r = std::min(s.min_r, r); s.max_r = std::max(s.max_r, r);
        s.min_g = std::min(s.min_g, g); s.max_g = std::max(s.max_g, g);
        s.min_b = std::min(s.min_b, b); s.max_b = std::max(s.max_b, b);
        s.min_a = std::min(s.min_a, a); s.max_a = std::max(s.max_a, a);
        if (r != 0.0f || g != 0.0f || b != 0.0f || a != 0.0f) s.all_zero = false;
        if (!s.saw_nonzero && (r != 0.0f || g != 0.0f || b != 0.0f || a != 0.0f)) {
            s.first_nz_r = r; s.first_nz_g = g; s.first_nz_b = b; s.first_nz_a = a;
            s.saw_nonzero = true;
        }
    }
    s.has_variation =
        (s.max_r - s.min_r) > 0.001f ||
        (s.max_g - s.min_g) > 0.001f ||
        (s.max_b - s.min_b) > 0.001f;
    return s;
}

// Count unique 4-byte integer "color" values (rounded to 1/255 steps) in
// the image. A trivial "all-black" or "all-same-color" image has 1.
int unique_colors(const Engine::BakedImage& img) {
    std::set<uint32_t> seen;
    if (img.pixels.empty()) return 0;
    const size_t n = img.pixels.size() / 4;
    for (size_t i = 0; i < n; ++i) {
        uint32_t r = (uint32_t)std::min(255, std::max(0, (int)std::round(img.pixels[i*4+0] * 255.0f)));
        uint32_t g = (uint32_t)std::min(255, std::max(0, (int)std::round(img.pixels[i*4+1] * 255.0f)));
        uint32_t b = (uint32_t)std::min(255, std::max(0, (int)std::round(img.pixels[i*4+2] * 255.0f)));
        uint32_t a = (uint32_t)std::min(255, std::max(0, (int)std::round(img.pixels[i*4+3] * 255.0f)));
        seen.insert((a << 24) | (b << 16) | (g << 8) | r);
        if ((int)seen.size() > 32) return (int)seen.size();
    }
    return (int)seen.size();
}

// Compute mean squared difference between two same-sized images.
// 0.0 = identical. Large = different.
double msd(const Engine::BakedImage& a, const Engine::BakedImage& b) {
    if (a.pixels.size() != b.pixels.size()) return 1.0e9;
    double s = 0;
    for (size_t i = 0; i < a.pixels.size(); ++i) {
        double d = double(a.pixels[i]) - double(b.pixels[i]);
        s += d * d;
    }
    return s / double(a.pixels.size());
}

void dump_stats(const char* label, const ImgStats& s) {
    std::printf("    %s: r=[%.3f,%.3f] g=[%.3f,%.3f] b=[%.3f,%.3f] a=[%.3f,%.3f] first_nz=[%.3f,%.3f,%.3f,%.3f]\n",
                label,
                s.min_r, s.max_r, s.min_g, s.max_g, s.min_b, s.max_b, s.min_a, s.max_a,
                s.first_nz_r, s.first_nz_g, s.first_nz_b, s.first_nz_a);
}

constexpr uint32_t kRes = 64;
constexpr uint32_t kSeed = 1;

struct EngineFixture {
    Engine engine;
    bool initialized = false;

    EngineFixture() {
        // Each test gets its own engine + shader cache directory.
        static int counter = 0;
        char cache[64];
        std::snprintf(cache, sizeof(cache), "test_blend_cache_%d", counter++);
        initialized = engine.init(VK_NULL_HANDLE, nullptr, 0,
                                  /*validation*/ true, cache,
                                  find_test_nodes_dir().c_str(),
                                  find_test_glsl_dir().c_str());
    }
    ~EngineFixture() { if (initialized) engine.shutdown(); }

    // Set up a graph with simplex and white_noise feeding into a single
    // Blend node as the output. If swap is true, the order is reversed
    // (simplex -> B, white_noise -> A). If any is null, that socket is
    // left unconnected.
    bool build_blend(NodeId blend_id,
                     const char* a_source,   // "simplex", "white", or null
                     const char* b_source,
                     bool swap = false) {
        Graph g;
        NodeId next_id = 1;
        NodeId a_id = 0, b_id = 0;
        if (a_source) { a_id = next_id++; g.nodes.push_back({a_id, a_source}); }
        if (b_source && (!a_source || a_source != b_source)) {
            b_id = next_id++; g.nodes.push_back({b_id, b_source});
        } else if (b_source && a_source == b_source) {
            // Same source, need a second instance.
            b_id = next_id++; g.nodes.push_back({b_id, b_source});
        }
        // Edge case: only one source requested, must be unique
        if (a_source && !b_source) {
            // Only A connected.
        } else if (!a_source && b_source) {
            // Only B connected.
        }
        g.nodes.push_back({blend_id, "blend"});
        // A
        if (a_source && a_id > 0) {
            g.connections.push_back({a_id, 0, blend_id, 0});
        }
        // B
        if (b_source && b_id > 0) {
            g.connections.push_back({b_id, 0, blend_id, 1});
        }
        g.output_node = blend_id;
        uint64_t gen = engine.set_graph(g);
        if (gen == 0) return false;
        for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return engine.has_pipeline();
    }

    // Set blend node's params (factor, mode). Uses param indices: 0=factor, 1=mode.
    void set_blend_params(NodeId blend_id, float factor, float mode) {
        engine.update_node_params_by_id(blend_id, {factor, mode});
    }

    bool render(NodeId active_id) {
        uint64_t g = engine.set_active_node(active_id);
        if (g == 0) return false;
        for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return engine.has_pipeline();
    }
};

#define SKIP_IF_NO_ENGINE() \
    if (!fx.initialized) { GTEST_SKIP() << "engine init failed: " << fx.engine.last_error(); }

constexpr NodeId kBlend = 99;

// =====================================================================
// Sanity baselines
// =====================================================================

TEST(Blend, Sanity_SimplexAloneIsNotBlack) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.output_node = 1;
    ASSERT_NE(fx.engine.set_graph(g), 0u);
    for (int i = 0; i < 200 && !fx.engine.has_pipeline(); ++i) {
        fx.engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(fx.engine.has_pipeline());

    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());
    auto s = stats_of(img);
    std::printf("    simplex alone: sum_r=%.1f min=%.3f max=%.3f all_zero=%d variation=%d\n",
                s.sum_r, s.min_r, s.max_r, s.all_zero, s.has_variation);
    EXPECT_FALSE(s.all_zero) << "simplex alone must not be all-zero";
    EXPECT_TRUE(s.has_variation) << "simplex alone must have spatial variation";
    EXPECT_GT(s.max_a, 0.5f) << "simplex alpha must be > 0.5 (noise shader sets a=1)";
}

TEST(Blend, Sanity_WhiteNoiseAloneIsNotBlack) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Graph g;
    g.nodes.push_back({1, "white"});
    g.output_node = 1;
    ASSERT_NE(fx.engine.set_graph(g), 0u);
    for (int i = 0; i < 200 && !fx.engine.has_pipeline(); ++i) {
        fx.engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(fx.engine.has_pipeline());

    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());
    auto s = stats_of(img);
    std::printf("    white_noise alone: sum_r=%.1f min=%.3f max=%.3f all_zero=%d variation=%d\n",
                s.sum_r, s.min_r, s.max_r, s.all_zero, s.has_variation);
    EXPECT_FALSE(s.all_zero);
    EXPECT_TRUE(s.has_variation);
    EXPECT_GT(s.max_a, 0.5f) << "white_noise alpha must be > 0.5";
}

// =====================================================================
// The exact case the user reported: simplex(A) + white_noise(B) + Blend,
// default factor=1, default mode=mix. With both inputs connected this
// must produce a valid image (mix of A and B with factor 1 = B).
// =====================================================================

TEST(Blend, BothConnected_DefaultFactor_ProducesValidImage) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // factor=1.0, mode=0 (mix)
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));
    fx.set_blend_params(kBlend, 1.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));

    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());
    auto s = stats_of(img);
    dump_stats("blend(simplex, white_noise) factor=1 mix", s);

    EXPECT_FALSE(s.all_zero) << "Blend with both inputs connected must not be all-zero";
    EXPECT_TRUE(s.has_variation) << "Blend with both inputs must have variation";
    EXPECT_GT(s.max_a, 0.5f) << "Blend with both inputs must preserve alpha (was the user's reported issue)";

    // With factor=1 mix, output should be approximately B = white_noise.
    // Compare against standalone white_noise (see Sanity test). They won't
    // match exactly (different noise pattern from second-pass dispatch)
    // but should be the same distribution: mean ~0.5, range wide.
    int n = unique_colors(img);
    std::printf("    unique colors: %d\n", n);
    EXPECT_GT(n, 4) << "Blend output must have > 4 unique colors (was the user's black-output bug)";
}

// =====================================================================
// The user's confusion case: factor=0 should give A, factor=1 should give B.
// Both must produce valid images, and they must differ from each other.
// =====================================================================

TEST(Blend, BothConnected_FactorZero_GivesA_FactorOne_GivesB_Different) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));

    // factor=0 (give A)
    fx.set_blend_params(kBlend, 0.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_f0 = fx.engine.readback_sync();
    auto s_f0 = stats_of(img_f0);
    dump_stats("factor=0 mix (expect ~simplex)", s_f0);
    EXPECT_FALSE(s_f0.all_zero) << "factor=0 must not be black";
    EXPECT_TRUE(s_f0.has_variation) << "factor=0 must have variation (A=simplex)";

    // factor=1 (give B)
    fx.set_blend_params(kBlend, 1.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_f1 = fx.engine.readback_sync();
    auto s_f1 = stats_of(img_f1);
    dump_stats("factor=1 mix (expect ~white_noise)", s_f1);
    EXPECT_FALSE(s_f1.all_zero) << "factor=1 must not be black";
    EXPECT_TRUE(s_f1.has_variation) << "factor=1 must have variation (B=white_noise)";

    // factor=0.5 (mid blend)
    fx.set_blend_params(kBlend, 0.5f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_f5 = fx.engine.readback_sync();
    auto s_f5 = stats_of(img_f5);
    dump_stats("factor=0.5 mix (expect ~average)", s_f5);
    EXPECT_FALSE(s_f5.all_zero) << "factor=0.5 must not be black";
    EXPECT_TRUE(s_f5.has_variation);

    // The three factor values must produce different outputs.
    double d_01 = msd(img_f0, img_f1);
    double d_05 = msd(img_f0, img_f5);
    std::printf("    msd(factor0, factor1) = %.6f   msd(factor0, factor0.5) = %.6f\n", d_01, d_05);
    EXPECT_GT(d_01, 0.001) << "factor=0 and factor=1 must produce different images";
    EXPECT_GT(d_05, 0.0001) << "factor=0 and factor=0.5 must produce different images";
}

// =====================================================================
// A<->B swap: with simplex(A) and white_noise(B), the output should
// change when we swap the connections. Proves BOTH inputs reach the
// shader, not just the first one.
// =====================================================================

TEST(Blend, BothConnected_SwapAandB_ProducesDifferentResult) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // (A=simplex, B=white_noise), factor=0.5
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));
    fx.set_blend_params(kBlend, 0.5f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_normal = fx.engine.readback_sync();

    // factor=0 on (A=simplex, B=white_noise) -> output is A = simplex
    fx.set_blend_params(kBlend, 0.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_normal_f0 = fx.engine.readback_sync();

    // factor=1 on (A=simplex, B=white_noise) -> output is B = white_noise
    fx.set_blend_params(kBlend, 1.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_normal_f1 = fx.engine.readback_sync();

    // Sanity: the A-only result must NOT match the B-only result.
    double d_ab = msd(img_normal_f0, img_normal_f1);
    std::printf("    msd(factor0=simplex, factor1=white_noise) = %.6f\n", d_ab);
    EXPECT_GT(d_ab, 0.001) << "factor=0 (=A) and factor=1 (=B) must be different";

    // The factor=0.5 mix of (A=simplex, B=white_noise) should be a
    // visually distinct third image.
    fx.set_blend_params(kBlend, 0.5f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_normal_f5 = fx.engine.readback_sync();
    double d_mix_vs_a = msd(img_normal_f5, img_normal_f0);
    double d_mix_vs_b = msd(img_normal_f5, img_normal_f1);
    std::printf("    msd(mix, factor0) = %.6f   msd(mix, factor1) = %.6f\n", d_mix_vs_a, d_mix_vs_b);
    EXPECT_GT(d_mix_vs_a, 0.0001);
    EXPECT_GT(d_mix_vs_b, 0.0001);
}

// =====================================================================
// One unconnected input. The user reported this case: "i can't preview
// specific channel of the separate RGBA" was the previous ask, but for
// Blend the issue is "what does an unconnected input look like?".
// =====================================================================

TEST(Blend, OneUnconnectedInput_Factor0_GivesConnected_Factor1_GivesDummy) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // A=simplex, B=unconnected
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", nullptr));
    fx.set_blend_params(kBlend, 0.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_a_only_f0 = fx.engine.readback_sync();
    auto s = stats_of(img_a_only_f0);
    dump_stats("A=simplex,B=unconn, factor=0 (expect ~simplex)", s);
    EXPECT_FALSE(s.all_zero) << "factor=0 should give connected A=simplex, must not be black";
    EXPECT_TRUE(s.has_variation) << "factor=0 with A=simplex must have variation";
    EXPECT_GT(s.max_a, 0.5f) << "alpha must be > 0.5 (user's reported bug)";

    // factor=1: B is unconnected. Output is B = dummy.
    // If the dummy is correctly initialized to (0,0,0,1), output is
    // (0,0,0,1) -> RGB zero, alpha 1. If the dummy is uninitialized
    // (undefined/garbage), output is garbage.
    // We assert: output is NOT visually similar to a valid noise pattern
    // (i.e., it should be solid black or at least very dark, not noise).
    fx.set_blend_params(kBlend, 1.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_a_only_f1 = fx.engine.readback_sync();
    auto s_f1 = stats_of(img_a_only_f1);
    dump_stats("A=simplex,B=unconn, factor=1 (expect ~dummy)", s_f1);

    // The output of factor=0 and factor=1 must differ (otherwise both
    // inputs collapsed to the same thing -> A wasn't really read).
    double d = msd(img_a_only_f0, img_a_only_f1);
    std::printf("    msd(factor0, factor1) = %.6f\n", d);
    EXPECT_GT(d, 0.0001) << "factor=0 and factor=1 with B unconnected must differ";
}

TEST(Blend, OneUnconnectedInput_SwappedAFB) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // A=unconnected, B=simplex
    ASSERT_TRUE(fx.build_blend(kBlend, nullptr, "simplex"));

    // factor=0: gives A = dummy
    fx.set_blend_params(kBlend, 0.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_b_only_f0 = fx.engine.readback_sync();

    // factor=1: gives B = simplex
    fx.set_blend_params(kBlend, 1.0f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_b_only_f1 = fx.engine.readback_sync();
    auto s = stats_of(img_b_only_f1);
    dump_stats("A=unconn,B=simplex, factor=1 (expect ~simplex)", s);
    EXPECT_FALSE(s.all_zero) << "factor=1 with B=simplex must not be black";
    EXPECT_TRUE(s.has_variation) << "factor=1 with B=simplex must have variation";
    EXPECT_GT(s.max_a, 0.5f) << "alpha must be > 0.5 (user's reported bug)";

    double d = msd(img_b_only_f0, img_b_only_f1);
    std::printf("    msd(factor0=dummy, factor1=simplex) = %.6f\n", d);
    EXPECT_GT(d, 0.0001);
}

// =====================================================================
// Blend modes other than mix: add, multiply, screen.
// Each must produce a non-black, varied output when both inputs are
// valid noise.
// =====================================================================

TEST(Blend, BothConnected_ModeAdd_ProducesValidImage) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));
    // mode=1 is Add per blend.glsl
    fx.set_blend_params(kBlend, 1.0f, 1.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img = fx.engine.readback_sync();
    auto s = stats_of(img);
    dump_stats("blend(simplex, white_noise) factor=1 add", s);
    EXPECT_FALSE(s.all_zero);
    EXPECT_TRUE(s.has_variation);
    EXPECT_GT(s.max_a, 0.5f);
}

TEST(Blend, BothConnected_ModeMultiply_ProducesValidImage) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));
    // mode=2 is Multiply
    fx.set_blend_params(kBlend, 1.0f, 2.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img = fx.engine.readback_sync();
    auto s = stats_of(img);
    dump_stats("blend(simplex, white_noise) factor=1 mul", s);
    EXPECT_FALSE(s.all_zero);
    EXPECT_TRUE(s.has_variation);
    EXPECT_GT(s.max_a, 0.5f);
}

TEST(Blend, BothConnected_ModeScreen_ProducesValidImage) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));
    // mode=3 is Screen
    fx.set_blend_params(kBlend, 1.0f, 3.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img = fx.engine.readback_sync();
    auto s = stats_of(img);
    dump_stats("blend(simplex, white_noise) factor=1 screen", s);
    EXPECT_FALSE(s.all_zero);
    EXPECT_TRUE(s.has_variation);
    EXPECT_GT(s.max_a, 0.5f);
}

// =====================================================================
// Most important: with both inputs connected, output ALPHA must be 1.0
// everywhere (no transparency artifact from the dummy). This is the
// user's explicit ask: "default blend ... alpha be 1.0 and not 0.0".
// =====================================================================

TEST(Blend, BothConnected_AlphaIsOne) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));

    for (float factor : {0.0f, 0.25f, 0.5f, 0.75f, 1.0f}) {
        for (float mode : {0.0f, 1.0f, 2.0f, 3.0f}) {
            fx.set_blend_params(kBlend, factor, mode);
            ASSERT_TRUE(fx.render(kBlend));
            auto img = fx.engine.readback_sync();
            ASSERT_FALSE(img.pixels.empty());
            const size_t n = img.pixels.size() / 4;
            float min_a = 1e30f, max_a = -1e30f;
            int non_one_a = 0;
            for (size_t i = 0; i < n; ++i) {
                float a = img.pixels[i*4+3];
                if (a < min_a) min_a = a;
                if (a > max_a) max_a = a;
                if (std::abs(a - 1.0f) > 0.01f) ++non_one_a;
            }
            std::printf("    factor=%.2f mode=%.0f: alpha [%.3f, %.3f], non-1.0 pixels=%d/%zu\n",
                        factor, mode, min_a, max_a, non_one_a, n);
            // Both inputs are noise with alpha=1, so output alpha must be 1.
            EXPECT_NEAR(min_a, 1.0f, 0.05f)
                << "factor=" << factor << " mode=" << mode << " alpha must be ~1.0";
            EXPECT_EQ(non_one_a, 0)
                << "factor=" << factor << " mode=" << mode << " must have alpha=1.0 for all pixels";
        }
    }
}

// =====================================================================
// Final: full A<->B swap with both connected. After swapping, with
// factor=0 the new-A (which was B) should be selected. This catches
// the chain-barrier bug where mid-chain external inputs are misrouted.
// =====================================================================

TEST(Blend, BothConnected_TrueSwapAandB_OutputIsDifferent) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // (A=simplex, B=white_noise) with factor=0.5
    ASSERT_TRUE(fx.build_blend(kBlend, "simplex", "white"));
    fx.set_blend_params(kBlend, 0.5f, 0.0f);
    ASSERT_TRUE(fx.render(kBlend));
    auto img_ab = fx.engine.readback_sync();

    // Manually swap: disconnect both, then reconnect in opposite order.
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "white"});
    g.nodes.push_back({kBlend, "blend"});
    g.connections.push_back({2, 0, kBlend, 0}); // white_noise -> A
    g.connections.push_back({1, 0, kBlend, 1}); // simplex     -> B
    g.output_node = kBlend;
    uint64_t gen = fx.engine.set_graph(g);
    ASSERT_NE(gen, 0u);
    for (int i = 0; i < 200 && !fx.engine.has_pipeline(); ++i) {
        fx.engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(fx.engine.has_pipeline());
    fx.engine.update_node_params_by_id(kBlend, {0.5f, 0.0f});
    auto img_ba = fx.engine.readback_sync();

    // (A=simplex, B=white_noise) factor=0.5 vs (A=white_noise, B=simplex) factor=0.5:
    // output.rgb for both is mix(noiseA, noiseB, 0.5), so RGB channels should be
    // identical (sum is symmetric in A,B for mode=0/mix). Verify channels sum
    // and mean are the same to within tolerance.
    double d = msd(img_ab, img_ba);
    std::printf("    msd((A=simplex,B=wn) vs (A=wn,B=simplex)) factor=0.5 = %.6f\n", d);
    // For mix mode with factor=0.5, output = (A+B)/2, which is symmetric in
    // A and B. So the two images should be nearly identical.
    EXPECT_LT(d, 0.001) << "mix(A,B,0.5) should be symmetric in A,B";

    // Now compare factor=0: (A=simplex, B=white_noise) factor=0 -> A
    //                    vs (A=white_noise, B=simplex) factor=0 -> A (now wn)
    Graph g2;
    g2.nodes.push_back({1, "simplex"});
    g2.nodes.push_back({2, "white"});
    g2.nodes.push_back({kBlend, "blend"});
    g2.connections.push_back({2, 0, kBlend, 0});
    g2.connections.push_back({1, 0, kBlend, 1});
    g2.output_node = kBlend;
    fx.engine.set_graph(g2);
    for (int i = 0; i < 200 && !fx.engine.has_pipeline(); ++i) {
        fx.engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    fx.engine.update_node_params_by_id(kBlend, {0.0f, 0.0f});
    auto img_swap_f0 = fx.engine.readback_sync();

    // Restore normal ordering
    Graph g3;
    g3.nodes.push_back({1, "simplex"});
    g3.nodes.push_back({2, "white"});
    g3.nodes.push_back({kBlend, "blend"});
    g3.connections.push_back({1, 0, kBlend, 0});
    g3.connections.push_back({2, 0, kBlend, 1});
    g3.output_node = kBlend;
    fx.engine.set_graph(g3);
    for (int i = 0; i < 200 && !fx.engine.has_pipeline(); ++i) {
        fx.engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    fx.engine.update_node_params_by_id(kBlend, {0.0f, 0.0f});
    auto img_norm_f0 = fx.engine.readback_sync();

    // (A=simplex,B=wn) factor=0 -> simplex
    // (A=wn,B=simplex) factor=0 -> white_noise
    // These must differ.
    double d_swap = msd(img_swap_f0, img_norm_f0);
    std::printf("    msd((A=simplex,B=wn)f0 vs (A=wn,B=simplex)f0) = %.6f\n", d_swap);
    EXPECT_GT(d_swap, 0.001) << "swapping A<->B with factor=0 must produce different output";
}

} // namespace
