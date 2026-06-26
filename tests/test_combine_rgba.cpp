#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
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
    float  avg_r = 0, avg_g = 0, avg_b = 0, avg_a = 0;
    bool   all_zero = true;
    bool   has_variation = false;
    size_t count = 0;
};

ImgStats stats_of(const std::vector<float>& px) {
    ImgStats s;
    if (px.empty()) return s;
    const size_t n = px.size() / 4;
    s.count = n;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        float r = px[i+0], g = px[i+1], b = px[i+2], a = px[i+3];
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
    s.avg_r = (float)(s.sum_r / n);
    s.avg_g = (float)(s.sum_g / n);
    s.avg_b = (float)(s.sum_b / n);
    s.avg_a = (float)(s.sum_a / n);
    return s;
}

ImgStats stats_of_img(const Engine::BakedImage& img) {
    return stats_of(img.pixels);
}

double msd(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1.0e9;
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = double(a[i]) - double(b[i]);
        s += d * d;
    }
    return s / double(a.size());
}

double channel_msd(const std::vector<float>& a, const std::vector<float>& b, int ch) {
    if (a.size() != b.size()) return 1.0e9;
    double s = 0;
    size_t n = 0;
    for (size_t i = ch; i + 3 < a.size(); i += 4) {
        double d = double(a[i]) - double(b[i]);
        s += d * d;
        ++n;
    }
    return n > 0 ? s / double(n) : 0;
}

void dump_stats(const char* label, const ImgStats& st) {
    std::printf("    %s: r=[%.3f,%.3f avg=%.3f] g=[%.3f,%.3f avg=%.3f] b=[%.3f,%.3f avg=%.3f] a=[%.3f,%.3f avg=%.3f]\n",
                label, st.min_r, st.max_r, st.avg_r,
                st.min_g, st.max_g, st.avg_g,
                st.min_b, st.max_b, st.avg_b,
                st.min_a, st.max_a, st.avg_a);
}

constexpr uint32_t kRes = 64;

// ============================================================================
// Part 1: GLSL emission tests (no Engine required)
// ============================================================================

class CombineRGBAGLSL : public ::testing::Test {
protected:
    NodeLibrary lib;

    void SetUp() override {
        std::string err;
        int n = NodeRegistryLoader::load_from_directory(
            lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
        ASSERT_GT(n, 0) << "failed to load real nodes: " << err;
    }

    CompileGraphResult compile(Graph& g) {
        auto r = validate_graph(g, lib);
        EXPECT_TRUE(r.success) << r.error;
        if (!r.success) return {};
        return FusedGraphCompiler::compile(r.ir, lib, g.output_node);
    }
};

TEST_F(CombineRGBAGLSL, AllFourInputsConnected) {
    // noise1->R, noise2->G, noise3->B, noise4->A — all connected, no constants
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "perlin"});
    g.nodes.push_back({4, "perlin"});
    g.nodes.push_back({5, "combine_rgba"});
    g.connections.push_back({1, 0, 5, 0});
    g.connections.push_back({2, 0, 5, 1});
    g.connections.push_back({3, 0, 5, 2});
    g.connections.push_back({4, 0, 5, 3});
    g.output_node = 5;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_combine_rgba"), std::string::npos);
    EXPECT_NE(glsl.find("node_perlin"), std::string::npos);
    // All connected — no external input declarations
    EXPECT_EQ(glsl.find("_in_"), std::string::npos)
        << "all inputs connected: no external input declarations";
}

TEST_F(CombineRGBAGLSL, OneInputConnected_ROnly) {
    // perlin->R, G/B/A unconnected → baked vec4(0) for G, B, A
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "combine_rgba"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_combine_rgba"), std::string::npos);
    EXPECT_NE(glsl.find("node_perlin"), std::string::npos);
    // Unconnected sockets baked as vec4(0)
    EXPECT_NE(glsl.find("vec4(0"), std::string::npos)
        << "unconnected Vec4 must be baked as vec4(0)";
    EXPECT_EQ(glsl.find("_in_"), std::string::npos)
        << "no external input declarations for baked constants";
}

TEST_F(CombineRGBAGLSL, TwoInputsConnected_RG) {
    // perlin->R, perlin->G, B/A unconnected → baked vec4(0) for B, A
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 3, 1});
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_combine_rgba"), std::string::npos);
    EXPECT_NE(glsl.find("vec4(0"), std::string::npos)
        << "unconnected B/A must be baked as vec4(0)";
}

TEST_F(CombineRGBAGLSL, ThreeInputsConnected_RGB) {
    // perlin->R, perlin->G, perlin->B, A unconnected → baked vec4(0) for A
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "perlin"});
    g.nodes.push_back({4, "combine_rgba"});
    g.connections.push_back({1, 0, 4, 0});
    g.connections.push_back({2, 0, 4, 1});
    g.connections.push_back({3, 0, 4, 2});
    g.output_node = 4;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_combine_rgba"), std::string::npos);
    EXPECT_NE(glsl.find("vec4(0"), std::string::npos)
        << "unconnected A must be baked as vec4(0)";
}

TEST_F(CombineRGBAGLSL, ZeroInputs_AllUnconnected) {
    // All 4 sockets unconnected → R/G/B baked as vec4(0), A baked as vec4(1)
    Graph g;
    g.nodes.push_back({1, "combine_rgba"});
    g.output_node = 1;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_combine_rgba"), std::string::npos);
    // All unconnected → baked constants, no _in_ externals
    EXPECT_EQ(glsl.find("_in_"), std::string::npos)
        << "all inputs unconnected: no external declarations (all baked)";
    EXPECT_NE(glsl.find("vec4(0"), std::string::npos)
        << "all unconnected sockets must be baked as vec4(0)";
}

// ============================================================================
// Part 2: Pixel correctness tests (Engine required)
// ============================================================================

class CombineRGBAPixel : public ::testing::Test {
protected:
    Engine engine;

    void SetUp() override {
        bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                              "test_combine_rgba_pixel",
                              find_test_nodes_dir().c_str(),
                              find_test_glsl_dir().c_str());
        if (!ok) GTEST_SKIP() << engine.last_error();
    }

    void TearDown() override {
        engine.shutdown();
    }

    std::vector<float> render_and_readback(uint64_t gen, NodeId active_id) {
        PushConstants pc{};
        pc.resolution_x = kRes; pc.resolution_y = kRes;
        pc.seed = 1; pc.time = 0.0f;
        uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
        if (ticket == 0) return {};
        std::vector<float> px;
        uint32_t w = 0, h = 0;
        uint64_t og = 0;
        for (int i = 0; i < 300; ++i) {
            if (engine.async_readback().poll(engine.ctx(), px, w, h, og)) return px;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return {};
    }
};

TEST_F(CombineRGBAPixel, AllFourInputs_4DifferentNoises) {
    // 4 independent perlin nodes → R, G, B, A channels
    // Each channel should be a different noise field
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "perlin"});
    g.nodes.push_back({4, "perlin"});
    g.nodes.push_back({5, "combine_rgba"});
    g.connections.push_back({1, 0, 5, 0});
    g.connections.push_back({2, 0, 5, 1});
    g.connections.push_back({3, 0, 5, 2});
    g.connections.push_back({4, 0, 5, 3});
    g.output_node = 5;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 5);
    ASSERT_FALSE(px.empty());
    auto s = stats_of(px);
    dump_stats("4-noise combine", s);

    EXPECT_FALSE(s.all_zero) << "output must not be black";
    EXPECT_TRUE(s.has_variation) << "output must have spatial variation";

    // All 4 channels should have non-zero values
    EXPECT_GT(s.avg_r, 0.0f) << "R channel must be non-zero";
    EXPECT_GT(s.avg_g, 0.0f) << "G channel must be non-zero";
    EXPECT_GT(s.avg_b, 0.0f) << "B channel must be non-zero";
    EXPECT_GT(s.avg_a, 0.0f) << "A channel must be non-zero";
}

TEST_F(CombineRGBAPixel, OneInput_ROnly_NoiseInR) {
    // perlin → R only, G/B/A unconnected
    // Mono dummy = (1,0,0,1) → G=1.0, B=1.0, A=1.0
    // R channel should vary (noise), G/B/A should be constant ~1.0
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "combine_rgba"});
    g.connections.push_back({1, 0, 2, 0});  // perlin → R (socket 0)
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 2);
    ASSERT_FALSE(px.empty());
    auto s = stats_of(px);
    dump_stats("1-input (R) combine", s);

    // R channel: should have variation (perlin noise)
    EXPECT_TRUE(s.has_variation) << "R channel must have variation from perlin";

    // G, B, A: unconnected Vec4 → baked vec4(0.0)
    EXPECT_NEAR(s.avg_g, 0.0f, 0.01f) << "G channel must be 0.0 (baked constant)";
    EXPECT_NEAR(s.avg_b, 0.0f, 0.01f) << "B channel must be 0.0 (baked constant)";
}

TEST_F(CombineRGBAPixel, TwoInputs_RG_NoiseInRG) {
    // perlin → R, perlin → G, B/A unconnected → baked vec4(0.0)
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 3, 1});
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 3);
    ASSERT_FALSE(px.empty());
    auto s = stats_of(px);
    dump_stats("2-input (RG) combine", s);

    float r_range = s.max_r - s.min_r;
    float g_range = s.max_g - s.min_g;
    EXPECT_GT(r_range, 0.001f) << "R channel must have variation";
    EXPECT_GT(g_range, 0.001f) << "G channel must have variation";

    // B, A: unconnected → baked vec4(0.0)
    EXPECT_NEAR(s.avg_b, 0.0f, 0.01f) << "B channel must be 0.0 (baked constant)";
}

TEST_F(CombineRGBAPixel, ThreeInputs_RGB_NoiseInRGB) {
    // perlin → R, perlin → G, perlin → B, A unconnected → baked vec4(0.0)
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "perlin"});
    g.nodes.push_back({4, "combine_rgba"});
    g.connections.push_back({1, 0, 4, 0});
    g.connections.push_back({2, 0, 4, 1});
    g.connections.push_back({3, 0, 4, 2});
    g.output_node = 4;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 4);
    ASSERT_FALSE(px.empty());
    auto s = stats_of(px);
    dump_stats("3-input (RGB) combine", s);

    float r_range = s.max_r - s.min_r;
    float g_range = s.max_g - s.min_g;
    float b_range = s.max_b - s.min_b;
    EXPECT_GT(r_range, 0.001f) << "R channel must vary";
    EXPECT_GT(g_range, 0.001f) << "G channel must vary";
    EXPECT_GT(b_range, 0.001f) << "B channel must vary";

    // A: unconnected → baked vec4(0.0)
    EXPECT_NEAR(s.avg_a, 1.0f, 0.01f) << "A channel must be 1.0 (baked default)";
}

TEST_F(CombineRGBAPixel, ZeroInputs_AllDummy) {
    // combine_rgba alone, no connections → all channels baked as vec4(0.0)
    Graph g;
    g.nodes.push_back({1, "combine_rgba"});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 1);
    ASSERT_FALSE(px.empty());
    auto s = stats_of(px);
    dump_stats("0-input combine (all baked vec4(0))", s);

    // All channels baked constant, no dummy texture. A defaults to 1.0.
    EXPECT_NEAR(s.avg_r, 0.0f, 0.01f) << "R must be 0.0";
    EXPECT_NEAR(s.avg_g, 0.0f, 0.01f) << "G must be 0.0";
    EXPECT_NEAR(s.avg_b, 0.0f, 0.01f) << "B must be 0.0";
    EXPECT_NEAR(s.avg_a, 1.0f, 0.01f) << "A must be 1.0 (default)";
    EXPECT_FALSE(s.has_variation) << "all-baked output should be uniform";
}

TEST_F(CombineRGBAPixel, SeparateCombine_Roundtrip) {
    // simplex → separate_rgba → combine_rgba
    // The roundtrip should reconstruct the original noise channels
    // separate splits: R=vec4(r,r,r,1), G=vec4(g,g,g,1), B=vec4(b,b,b,1), A=vec4(a,a,a,1)
    // combine reconstructs: vec4(R.r, G.r, B.r, A.r) = vec4(r, g, b, a)
    // So combine_rgba(separate_rgba(x)) should equal x

    // Reference: standalone simplex
    Graph g_ref;
    g_ref.nodes.push_back({1, "simplex"});
    g_ref.output_node = 1;

    uint64_t gen_ref = engine.set_graph(g_ref);
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto px_ref = render_and_readback(gen_ref, 1);
    ASSERT_FALSE(px_ref.empty());

    // Roundtrip: simplex → separate → combine
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "separate_rgba"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 2, 0});  // simplex → separate
    g.connections.push_back({2, 0, 3, 0});  // separate.R → combine.R
    g.connections.push_back({2, 1, 3, 1});  // separate.G → combine.G
    g.connections.push_back({2, 2, 3, 2});  // separate.B → combine.B
    g.connections.push_back({2, 3, 3, 3});  // separate.A → combine.A
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 3);
    ASSERT_FALSE(px.empty());

    double diff = msd(px_ref, px);
    std::printf("    msd(simplex, combine(separate(simplex))) = %.6f\n", diff);
    EXPECT_LT(diff, 0.001) << "separate→combine roundtrip should reconstruct original";
}

TEST_F(CombineRGBAPixel, SeparateCombine_SingleChannel) {
    // simplex → separate → only R connected to combine, G/B/A unconnected → baked vec4(0)
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "separate_rgba"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 3);
    ASSERT_FALSE(px.empty());
    auto s = stats_of(px);
    dump_stats("separate→combine (R only)", s);

    // R varies (noise), G/B = 0 (baked), A = 1.0 (default)
    EXPECT_TRUE(s.has_variation) << "R must vary with noise";
    EXPECT_NEAR(s.avg_g, 0.0f, 0.01f) << "G must be 0.0";
    EXPECT_NEAR(s.avg_b, 0.0f, 0.01f) << "B must be 0.0";
    EXPECT_NEAR(s.avg_a, 1.0f, 0.01f) << "A must be 1.0 (default)";
}

TEST_F(CombineRGBAPixel, TwoSeperateSources_RG) {
    // simplex → R, perlin → G, B/A unconnected → baked vec4(0.0)
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "combine_rgba"});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 3, 1});
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto px = render_and_readback(gen, 3);
    ASSERT_FALSE(px.empty());
    auto s = stats_of(px);
    dump_stats("2-source (simplex+perlin) combine", s);

    float r_range = s.max_r - s.min_r;
    float g_range = s.max_g - s.min_g;
    EXPECT_GT(r_range, 0.001f) << "R channel must vary (simplex)";
    EXPECT_GT(g_range, 0.001f) << "G channel must vary (perlin)";

    // B/A: unconnected → baked vec4(0.0)
    EXPECT_NEAR(s.avg_b, 0.0f, 0.01f) << "B must be 0.0 (baked constant)";
}

} // namespace
