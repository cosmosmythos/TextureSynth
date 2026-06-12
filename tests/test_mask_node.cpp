#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "test_assets.hpp"
#include <cmath>
#include <cstdio>
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
    }
    s.has_variation =
        (s.max_r - s.min_r) > 0.001f ||
        (s.max_g - s.min_g) > 0.001f ||
        (s.max_b - s.min_b) > 0.001f;
    return s;
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

void dump_stats(const char* label, const ImgStats& st) {
    std::printf("    %s: r=[%.3f,%.3f] g=[%.3f,%.3f] b=[%.3f,%.3f] a=[%.3f,%.3f]\n",
                label, st.min_r, st.max_r, st.min_g, st.max_g, st.min_b, st.max_b, st.min_a, st.max_a);
}

constexpr uint32_t kRes = 64;

struct EngineFixture {
    Engine engine;
    bool initialized = false;

    EngineFixture() {
        static int counter = 0;
        char cache[64];
        std::snprintf(cache, sizeof(cache), "test_mask_cache_%d", counter++);
        initialized = engine.init(VK_NULL_HANDLE, nullptr, 0,
                                  true, cache,
                                  find_test_nodes_dir().c_str(),
                                  find_test_glsl_dir().c_str());
    }
    ~EngineFixture() { if (initialized) engine.shutdown(); }

    bool build_graph(const Graph& g) {
        uint64_t gen = engine.set_graph(g);
        if (gen == 0) return false;
        for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return engine.has_pipeline();
    }

    bool render(NodeId active_id, int max_retries = 800) {
        uint64_t g = engine.set_active_node(active_id);
        if (g == 0) return false;
        for (int i = 0; i < max_retries && !engine.has_pipeline(); ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return engine.has_pipeline();
    }
};

#define SKIP_IF_NO_ENGINE() \
    if (!fx.initialized) { GTEST_SKIP() << "engine init failed: " << fx.engine.last_error(); }

// =====================================================================
    // Invert mask tests
// Inputs: [0]=mask, [1]=color
// Params: (none)
// =====================================================================

TEST(Mask, Invert_Mask1_FullyInverts) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // simplex -> color (input[1])
    g.output_node = 2;
    ASSERT_TRUE(fx.build_graph(g));

    // mask=1.0 (default slider value via SSBO)
    fx.engine.update_node_params_by_id(2, {1.0f});
    ASSERT_TRUE(fx.render(2));
    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());
    auto s = stats_of(img);
    dump_stats("invert(simplex) mask=1", s);
    EXPECT_FALSE(s.all_zero) << "invert with mask=1 must not be black";
    EXPECT_TRUE(s.has_variation) << "invert with mask=1 must have variation";
    EXPECT_GT(s.max_a, 0.5f) << "alpha must be > 0.5";
}

TEST(Mask, Invert_Mask0_PassthroughInput) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // Standalone simplex reference
    Graph g_ref;
    g_ref.nodes.push_back({1, "simplex"});
    g_ref.output_node = 1;
    ASSERT_TRUE(fx.build_graph(g_ref));
    ASSERT_TRUE(fx.render(1));
    auto ref_img = fx.engine.readback_sync();
    ASSERT_FALSE(ref_img.pixels.empty());

    // simplex -> invert with mask=0
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // simplex -> color (input[1])
    g.output_node = 2;
    ASSERT_TRUE(fx.build_graph(g));

    // mask=0.0 -> output should be input (passthrough)
    fx.engine.update_node_params_by_id(2, {0.0f});
    ASSERT_TRUE(fx.render(2));
    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());

    double diff = msd(ref_img, img);
    std::printf("    msd(simplex, invert(simplex) mask=0) = %.6f\n", diff);
    EXPECT_LT(diff, 0.001) << "invert with mask=0 should passthrough input unchanged";
}

TEST(Mask, Invert_Mask0and1_Different) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // simplex -> color (input[1])
    g.output_node = 2;
    ASSERT_TRUE(fx.build_graph(g));

    // mask=0
    fx.engine.update_node_params_by_id(2, {0.0f});
    ASSERT_TRUE(fx.render(2));
    auto img_m0 = fx.engine.readback_sync();

    // mask=1
    fx.engine.update_node_params_by_id(2, {1.0f});
    ASSERT_TRUE(fx.render(2));
    auto img_m1 = fx.engine.readback_sync();

    double diff = msd(img_m0, img_m1);
    std::printf("    msd(mask=0, mask=1) = %.6f\n", diff);
    EXPECT_GT(diff, 0.001) << "invert mask=0 and mask=1 must produce different results";
}

// =====================================================================
    // Grayscale mask tests
// Inputs: [0]=mask, [1]=color
// Params: [0]=mode
// =====================================================================

TEST(Mask, Grayscale_Mask1_GrayscaleOutput) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "grayscale"});
    g.connections.push_back({1, 0, 2, 1});  // simplex -> color (input[1])
    g.output_node = 2;
    ASSERT_TRUE(fx.build_graph(g));

    // mode=0 (luminance), mask=1.0
    fx.engine.update_node_params_by_id(2, {0.0f, 1.0f});
    ASSERT_TRUE(fx.render(2));
    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());
    auto s = stats_of(img);
    dump_stats("grayscale(simplex) mask=1", s);
    EXPECT_FALSE(s.all_zero);
    EXPECT_TRUE(s.has_variation);

    // For luminance grayscale, each pixel should have R=G=B
    const size_t n = img.pixels.size() / 4;
    int non_gray = 0;
    for (size_t i = 0; i < n; ++i) {
        float r = img.pixels[i*4+0];
        float gg = img.pixels[i*4+1];
        float b = img.pixels[i*4+2];
        if (std::abs(r - gg) > 0.01f || std::abs(gg - b) > 0.01f) ++non_gray;
    }
    std::printf("    non-gray pixels: %d / %zu\n", non_gray, n);
    EXPECT_EQ(non_gray, 0) << "grayscale output must have R=G=B for all pixels";
}

TEST(Mask, Grayscale_Mask0_PassthroughInput) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // Standalone simplex reference
    Graph g_ref;
    g_ref.nodes.push_back({1, "simplex"});
    g_ref.output_node = 1;
    ASSERT_TRUE(fx.build_graph(g_ref));
    ASSERT_TRUE(fx.render(1));
    auto ref_img = fx.engine.readback_sync();

    // simplex -> grayscale with mask=0
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "grayscale"});
    g.connections.push_back({1, 0, 2, 1});  // simplex -> color (input[1])
    g.output_node = 2;
    ASSERT_TRUE(fx.build_graph(g));
    fx.engine.update_node_params_by_id(2, {0.0f, 0.0f});
    ASSERT_TRUE(fx.render(2));
    auto img = fx.engine.readback_sync();

    double diff = msd(ref_img, img);
    std::printf("    msd(simplex, grayscale(simplex) mask=0) = %.6f\n", diff);
    EXPECT_LT(diff, 0.001) << "grayscale with mask=0 should passthrough input unchanged";
}

// =====================================================================
    // Blend mask tests
// Inputs: [0]=mask, [1]=a, [2]=b
// Params: [0]=mode
// =====================================================================

TEST(Mask, Blend_Mask0_PassthroughA) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // Standalone simplex reference
    Graph g_ref;
    g_ref.nodes.push_back({1, "simplex"});
    g_ref.output_node = 1;
    ASSERT_TRUE(fx.build_graph(g_ref));
    ASSERT_TRUE(fx.render(1));
    auto ref_img = fx.engine.readback_sync();

    // simplex -> A, white -> B, mask=0 (mix mode=0) -> output = A = simplex
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "white"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});  // simplex -> a (input[1])
    g.connections.push_back({2, 0, 3, 2});  // white   -> b (input[2])
    g.output_node = 3;
    ASSERT_TRUE(fx.build_graph(g));

    // mode=0 (mix), mask=0.0 -> output should be A (= simplex)
    fx.engine.update_node_params_by_id(3, {0.0f, 0.0f});
    ASSERT_TRUE(fx.render(3));
    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());

    double diff = msd(ref_img, img);
    std::printf("    msd(simplex, blend(simplex,white) mask=0 mix) = %.6f\n", diff);
    EXPECT_LT(diff, 0.001) << "blend with mask=0 should give A (unblended)";
}

TEST(Mask, Blend_Mask1_FullBlend) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // Standalone white_noise reference
    Graph g_ref;
    g_ref.nodes.push_back({1, "white"});
    g_ref.output_node = 1;
    ASSERT_TRUE(fx.build_graph(g_ref));
    ASSERT_TRUE(fx.render(1));
    auto ref_img = fx.engine.readback_sync();

    // simplex -> A, white -> B, mask=1 (mix mode=0) -> output = B = white_noise
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "white"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});  // simplex -> a (input[1])
    g.connections.push_back({2, 0, 3, 2});  // white   -> b (input[2])
    g.output_node = 3;
    ASSERT_TRUE(fx.build_graph(g));

    // mode=0 (mix), mask=1.0 -> output should be B (= white_noise)
    fx.engine.update_node_params_by_id(3, {0.0f, 1.0f});
    ASSERT_TRUE(fx.render(3));
    auto img = fx.engine.readback_sync();
    ASSERT_FALSE(img.pixels.empty());

    double diff = msd(ref_img, img);
    std::printf("    msd(white, blend(simplex,white) mask=1 mix) = %.6f\n", diff);
    EXPECT_LT(diff, 0.01) << "blend with mask=1 mix should give B (white_noise)";
}

TEST(Mask, Blend_Mask0and1_Different) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "white"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});  // simplex -> a (input[1])
    g.connections.push_back({2, 0, 3, 2});  // white   -> b (input[2])
    g.output_node = 3;
    ASSERT_TRUE(fx.build_graph(g));

    // mask=0
    fx.engine.update_node_params_by_id(3, {0.0f, 0.0f});
    ASSERT_TRUE(fx.render(3));
    auto img_m0 = fx.engine.readback_sync();

    // mask=1
    fx.engine.update_node_params_by_id(3, {0.0f, 1.0f});
    ASSERT_TRUE(fx.render(3));
    auto img_m1 = fx.engine.readback_sync();

    double diff = msd(img_m0, img_m1);
    std::printf("    msd(blend mask=0, blend mask=1) = %.6f\n", diff);
    EXPECT_GT(diff, 0.001) << "blend mask=0 and mask=1 must produce different results";
}

// =====================================================================
// Mask texture path: color_const connected to mask input
// =====================================================================

TEST(Mask, Blend_MaskConnectedViaColorConst_RespondsToValue) {
    EngineFixture fx;
    SKIP_IF_NO_ENGINE();

    // color_const -> mask, simplex -> A, white -> B
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "white"});
    g.nodes.push_back({3, "color_const"});  // mask source
    g.nodes.push_back({4, "blend"});
    g.connections.push_back({1, 0, 4, 1});  // simplex -> a (input[1])
    g.connections.push_back({2, 0, 4, 2});  // white   -> b (input[2])
    g.connections.push_back({3, 0, 4, 0});  // color   -> mask (input[0])
    g.output_node = 4;
    ASSERT_TRUE(fx.build_graph(g));

    // mode=0 (mix), color_const params = [mode=0, r=0, g=0, b=0, a=1] → vec4(0,0,0,1) → mask=0
    fx.engine.update_node_params_by_id(3, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
    fx.engine.update_node_params_by_id(4, {0.0f, 0.0f});  // mode=mix, mask ignored (texture path)
    ASSERT_TRUE(fx.render(4));
    auto img_m0 = fx.engine.readback_sync();
    ASSERT_FALSE(img_m0.pixels.empty());

    // color_const params = [mode=0, r=1, g=1, b=1, a=1] → vec4(1,1,1,1) → mask=1
    fx.engine.update_node_params_by_id(3, {0.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    ASSERT_TRUE(fx.render(4));
    auto img_m1 = fx.engine.readback_sync();
    ASSERT_FALSE(img_m1.pixels.empty());

    // Also get a standalone white_noise reference for comparison
    Graph g_white;
    g_white.nodes.push_back({10, "white"});
    g_white.output_node = 10;
    ASSERT_TRUE(fx.build_graph(g_white));
    ASSERT_TRUE(fx.render(10));
    auto white_img = fx.engine.readback_sync();

    // mask=1 should be pure B (= white noise)
    double diff_m1_white = msd(img_m1, white_img);
    std::printf("    msd(blend(mask=1 via color_const), white) = %.6f\n", diff_m1_white);
    EXPECT_LT(diff_m1_white, 0.01) << "blend with mask=1 from texture should give B (white_noise)";

    // mask=0 and mask=1 must produce different results
    double diff_0_1 = msd(img_m0, img_m1);
    std::printf("    msd(blend(mask=0), blend(mask=1)) via texture = %.6f\n", diff_0_1);
    EXPECT_GT(diff_0_1, 0.001) << "blend mask=0 and mask=1 via texture must differ";
}

} // namespace
