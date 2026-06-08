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

struct ImgStats {
    double sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
    float  min_r = 1e30f, min_g = 1e30f, min_b = 1e30f, min_a = 1e30f;
    float  max_r = -1e30f, max_g = -1e30f, max_b = -1e30f, max_a = -1e30f;
    bool   all_zero = true;
    bool   has_variation = false;
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
        static int counter = 0;
        char cache[64];
        std::snprintf(cache, sizeof(cache), "test_rgba_cache_%d", counter++);
        initialized = engine.init(VK_NULL_HANDLE, nullptr, 0,
                                  true, cache,
                                  find_test_nodes_dir().c_str(),
                                  find_test_glsl_dir().c_str());
    }
    ~EngineFixture() { if (initialized) engine.shutdown(); }

    bool set_graph_wait(const Graph& g) {
        uint64_t gen = engine.set_graph(g);
        if (gen == 0) return false;
        for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return engine.has_pipeline();
    }

    bool set_active_wait(NodeId id) {
        uint64_t gen = engine.set_active_node(id);
        if (gen == 0) return false;
        for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return engine.has_pipeline();
    }
};

#define SKIP_IF_NO_ENGINE() \
    if (!fx.initialized) { GTEST_SKIP() << "engine init failed: " << fx.engine.last_error(); }

// =====================================================================
// Baselines: verify the source nodes produce valid data.
// =====================================================================

TEST(SplitCombine, Sanity_SimplexAloneIsNotBlack) {
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
    EXPECT_FALSE(s.all_zero);
    EXPECT_TRUE(s.has_variation);
    EXPECT_GT(s.max_a, 0.5f);
}

TEST(SplitCombine, Sanity_WhiteNoiseAloneIsNotBlack) {
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
    EXPECT_FALSE(s.all_zero);
    EXPECT_TRUE(s.has_variation);
    EXPECT_GT(s.max_a, 0.5f);
}

// =====================================================================
// Split RGBA: all 4 output sockets must produce valid non-black images.
// =====================================================================

TEST(SplitCombine, SplitRGBA_AllFourOutputsAreValid) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Engine::BakedImage imgs[4];
    for (uint32_t sock = 0; sock < 4; ++sock) {
        Graph g;
        g.nodes.push_back({1, "simplex"});
        g.nodes.push_back({2, "split_rgba"});
        g.connections.push_back({1, 0, 2, 0});
        g.output_node = 2;
        g.output_socket = sock;
        ASSERT_TRUE(fx.set_graph_wait(g));
        imgs[sock] = fx.engine.readback_sync();
        ASSERT_FALSE(imgs[sock].pixels.empty());
        auto s = stats_of(imgs[sock]);
        std::printf("    split_rgba output_socket=%u: ", sock);
        dump_stats("stats", s);
        EXPECT_FALSE(s.all_zero) << "split_rgba output socket " << sock << " must not be all-zero";
        if (sock < 3) {
            EXPECT_TRUE(s.has_variation) << "split_rgba output socket " << sock << " must have variation (RGB from simplex vary)";
        } else {
            // Socket 3 = alpha channel. Simplex outputs alpha=1.0 everywhere,
            // so the separated A channel is constant 1.0 — no variation expected.
            EXPECT_FALSE(s.all_zero) << "split_rgba output socket 3 (A) must have alpha=1.0, not zero";
        }
        EXPECT_GT(s.max_a, 0.5f) << "split_rgba output socket " << sock << " alpha must be > 0.5";
    }
}

// =====================================================================
// The four output sockets must differ from each other (proves the
// R/G/B/A selection is actually splitting channels, not just copying).
// =====================================================================

TEST(SplitCombine, SplitRGBA_FourOutputsDiffer) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Engine::BakedImage imgs[4];
    for (uint32_t sock = 0; sock < 4; ++sock) {
        Graph g;
        g.nodes.push_back({1, "simplex"});
        g.nodes.push_back({2, "split_rgba"});
        g.connections.push_back({1, 0, 2, 0});
        g.output_node = 2;
        g.output_socket = sock;
        ASSERT_TRUE(fx.set_graph_wait(g));
        imgs[sock] = fx.engine.readback_sync();
        ASSERT_FALSE(imgs[sock].pixels.empty());
    }

    // Each channel is independent per simplex.glsl (different seed offsets).
    // So split_rgba outputs R, G, B, A should all look different.
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            double d = msd(imgs[a], imgs[b]);
            std::printf("    msd(socket_%d, socket_%d) = %.6f\n", a, b, d);
            EXPECT_GT(d, 0.001)
                << "split_rgba outputs " << a << " and " << b << " must differ";
        }
    }
}

// =====================================================================
// Split alpha must be 1.0 everywhere (shader broadcasts with a=1.0).
// =====================================================================

TEST(SplitCombine, SplitRGBA_AlphaIsOne) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    for (uint32_t sock = 0; sock < 4; ++sock) {
        Graph g;
        g.nodes.push_back({1, "simplex"});
        g.nodes.push_back({2, "split_rgba"});
        g.connections.push_back({1, 0, 2, 0});
        g.output_node = 2;
        g.output_socket = sock;
        ASSERT_TRUE(fx.set_graph_wait(g));
        auto img = fx.engine.readback_sync();
        ASSERT_FALSE(img.pixels.empty());

        const size_t n = img.pixels.size() / 4;
        int non_one_a = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::abs(img.pixels[i*4+3] - 1.0f) > 0.01f) ++non_one_a;
        }
        std::printf("    socket=%u: non-1.0 alpha pixels=%d/%zu\n", sock, non_one_a, n);
        EXPECT_EQ(non_one_a, 0) << "split_rgba socket " << sock << " must have alpha=1.0 everywhere";
    }
}

// =====================================================================
// Combine RGBA sanity: simplex -> split_rgba -> combine_rgba.
// All 4 split outputs feed into the 4 combine inputs. Output must be
// non-black, varied, and have alpha=1.0.
// =====================================================================

TEST(SplitCombine, CombineRGBA_ValidOutput) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // simplex -> split_rgba -> combine_rgba
    // split.R (socket 0) -> combine.r (socket 0)
    // split.G (socket 1) -> combine.g (socket 1)
    // split.B (socket 2) -> combine.b (socket 2)
    // split.A (socket 3) -> combine.a (socket 3)
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "split_rgba"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({2, 1, 3, 1});
    g.connections.push_back({2, 2, 3, 2});
    g.connections.push_back({2, 3, 3, 3});
    g.output_node = 3;
    ASSERT_TRUE(fx.set_graph_wait(g));

    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());
    auto s = stats_of(img);
    dump_stats("simplex -> split -> combine", s);
    EXPECT_FALSE(s.all_zero) << "combined output must not be all-zero";
    EXPECT_TRUE(s.has_variation) << "combined output must have variation";
    EXPECT_GT(s.max_a, 0.5f) << "combined output alpha must be > 0.5";
}

TEST(SplitCombine, CombineRGBA_AlphaIsOne) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "split_rgba"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({2, 1, 3, 1});
    g.connections.push_back({2, 2, 3, 2});
    g.connections.push_back({2, 3, 3, 3});
    g.output_node = 3;
    ASSERT_TRUE(fx.set_graph_wait(g));

    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());

    const size_t n = img.pixels.size() / 4;
    int non_one_a = 0;
    for (size_t i = 0; i < n; ++i) {
        float a = img.pixels[i*4+3];
        if (std::abs(a - 1.0f) > 0.01f) ++non_one_a;
    }
    std::printf("    combine_rgba: non-1.0 alpha pixels=%d/%zu\n", non_one_a, n);
    EXPECT_EQ(non_one_a, 0) << "combine_rgba must have alpha=1.0 everywhere (fix for unconnected dummy data bug)";
}

// =====================================================================
// Round-trip: simplex -> split -> combine should equal original simplex.
// The split node broadcasts each channel to all RGB, so:
//   combine.r reads from split.R's r.r = simplex.r
//   combine.g reads from split.G's g.r = simplex.g
//   combine.b reads from split.B's b.r = simplex.b
// combine hardcodes alpha to 1.0, and simplex also outputs alpha=1.0,
// so the round-trip should be identical.
// =====================================================================

TEST(SplitCombine, RoundTrip_SplitThenCombine) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // Reference: simplex alone.
    Graph g_ref;
    g_ref.nodes.push_back({1, "simplex"});
    g_ref.output_node = 1;
    ASSERT_TRUE(fx.set_graph_wait(g_ref));
    auto img_ref = fx.engine.readback_sync();
    ASSERT_FALSE(img_ref.pixels.empty());

    // Round-trip: simplex -> split -> combine.
    Graph g_rt;
    g_rt.nodes.push_back({1, "simplex"});
    g_rt.nodes.push_back({2, "split_rgba"});
    g_rt.nodes.push_back({3, "combine_rgba"});
    g_rt.connections.push_back({1, 0, 2, 0});
    g_rt.connections.push_back({2, 0, 3, 0});
    g_rt.connections.push_back({2, 1, 3, 1});
    g_rt.connections.push_back({2, 2, 3, 2});
    g_rt.connections.push_back({2, 3, 3, 3});
    g_rt.output_node = 3;
    ASSERT_TRUE(fx.set_graph_wait(g_rt));
    auto img_rt = fx.engine.readback_sync();
    ASSERT_FALSE(img_rt.pixels.empty());

    double d = msd(img_ref, img_rt);
    std::printf("    msd(simplex, split->combine) = %.6f\n", d);
    auto s_ref = stats_of(img_ref);
    auto s_rt  = stats_of(img_rt);
    dump_stats("simplex alone", s_ref);
    dump_stats("split->combine", s_rt);
    // Both are the same noise, same resolution, same seed — output
    // should be pixel-identical (the round trip is a no-op for RGB).
    EXPECT_LT(d, 0.001) << "simplex -> split -> combine must match original simplex";
}

// =====================================================================
// Split RGBA with white_noise (different source) to prove node-agnostic.
// =====================================================================

TEST(SplitCombine, SplitRGBA_WithWhiteNoise_AllValid) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    for (uint32_t sock = 0; sock < 4; ++sock) {
        Graph g;
        g.nodes.push_back({1, "white"});
        g.nodes.push_back({2, "split_rgba"});
        g.connections.push_back({1, 0, 2, 0});
        g.output_node = 2;
        g.output_socket = sock;
        ASSERT_TRUE(fx.set_graph_wait(g));
        auto img = fx.engine.readback_sync();
        ASSERT_FALSE(img.pixels.empty());
        auto s = stats_of(img);
        dump_stats("white->split socket=0", s);
        EXPECT_FALSE(s.all_zero) << "white->split socket " << sock << " must not be all-zero";
        if (sock < 3) {
            EXPECT_TRUE(s.has_variation) << "white->split socket " << sock << " must have variation";
        }
    }
}

// =====================================================================
// Combine with only one input connected (R only). The rest read dummy
// data. Output must still be valid — R channel shows the source, G and B
// are dummy (possibly 0), alpha must be 1.0.
// =====================================================================

TEST(SplitCombine, CombineRGBA_OnlyRConnected_ValidOutput) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // simplex -> split -> combine.r only. combine.g, combine.b, combine.a
    // read from unconnected dummies.
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "split_rgba"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0}); // only R connected
    g.output_node = 3;
    ASSERT_TRUE(fx.set_graph_wait(g));

    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());
    auto s = stats_of(img);
    dump_stats("combine_rgba (only R connected)", s);

    // R channel should have the split output.
    EXPECT_GT(s.max_r, 0.1f) << "R channel must contain data from source";
    // Not all-zero overall.
    EXPECT_FALSE(s.all_zero) << "with R connected, output must not be all-zero";
    // Alpha must be 1.0 from the combine_rgba fix.
    EXPECT_GT(s.max_a, 0.5f) << "alpha must be > 0.5 (combine fix)";

    const size_t n = img.pixels.size() / 4;
    int non_one_a = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(img.pixels[i*4+3] - 1.0f) > 0.01f) ++non_one_a;
    }
    EXPECT_EQ(non_one_a, 0) << "all pixels must have alpha=1.0 even with unconnected inputs";
}

} // namespace
