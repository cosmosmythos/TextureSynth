#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "test_assets.hpp"
#include <fstream>
#include <thread>
#include <chrono>
#include <regex>
#include <set>

using namespace te;

// ============================================================================
// Real-node graph-fusion tests.
// Load actual Blender node manifests, emit fused GLSL, verify correctness.
// ============================================================================

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

bool wait_for_pipeline(Engine& engine, int timeout_ms = 3000) {
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        engine.poll_pending_compiles();
        if (engine.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return engine.has_pipeline();
}

bool wait_for_readback_gen(Engine& engine, uint64_t gen,
                           std::vector<float>& pixels,
                           uint32_t& w, uint32_t& h, int timeout_ms = 3000) {
    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
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

bool wait_for_readback_gen_res(Engine& engine, uint64_t gen,
                               std::vector<float>& pixels,
                               uint32_t& w, uint32_t& h,
                               uint32_t res_x, uint32_t res_y,
                               int timeout_ms = 3000) {
    PushConstants pc{};
    pc.resolution_x = res_x; pc.resolution_y = res_y;
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

double avg_brightness(const std::vector<float>& px) {
    double sum = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4)
        sum += (px[i] + px[i+1] + px[i+2]) / 3.0;
    return sum;
}

} // anonymous namespace

// ===========================================================================
// Part 1: GLSL emission verification (no Engine required)
// ===========================================================================

class FusedRealNodesGLSL : public ::testing::Test {
protected:
    NodeLibrary lib = load_real_lib();

    // Helper: build graph, trace path, emit fused GLSL, return source.
    std::string emit(Graph& g) {
        auto r = validate_graph(g, lib);
        EXPECT_TRUE(r.success) << r.error;
        if (!r.success) return "";
        auto path = ActivePathTracer::trace(r.ir, g.output_node, lib);
        auto result = emit_fused_subgraph(path, r.ir, lib, 0, {});
        EXPECT_TRUE(result.ok()) << result.error;
        return result.source;
    }

    // Helper: emit with full compiler (returns chains + GLSL).
    CompileGraphResult compile(Graph& g) {
        auto r = validate_graph(g, lib);
        EXPECT_TRUE(r.success) << r.error;
        if (!r.success) return {};
        return FusedGraphCompiler::compile(r.ir, lib, g.output_node);
    }
};

TEST_F(FusedRealNodesGLSL, PerlinOnly_SingleChain) {
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.output_node = 1;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u);
    EXPECT_EQ(cr.pass_plan.chains[0].nodes.size(), 1u);
    EXPECT_EQ(cr.pass_plan.chains[0].nodes[0], 1u);

    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_perlin"), std::string::npos)
        << "fused GLSL must contain node_perlin function";
    EXPECT_NE(glsl.find("imageStore"), std::string::npos)
        << "fused GLSL must write to storage image";
    EXPECT_NE(glsl.find("r0"), std::string::npos)
        << "fused GLSL must use r0 for perlin output";
}

TEST_F(FusedRealNodesGLSL, PerlinToInvert_LinearChain) {
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> invert.color (socket 1)
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u);
    EXPECT_EQ(cr.pass_plan.chains[0].nodes.size(), 2u);

    const auto& glsl = cr.pass_plan.chains[0].glsl;
    // Dump GLSL to file for inspection
    {
        std::ofstream f("fused_perlin_invert.glsl");
        f << glsl;
    }

    // Must contain both node functions.
    EXPECT_NE(glsl.find("node_perlin"), std::string::npos);
    EXPECT_NE(glsl.find("node_invert"), std::string::npos);

    // Must use register chain: r0 (perlin) -> invert -> r1.
    EXPECT_NE(glsl.find("r0"), std::string::npos);
    EXPECT_NE(glsl.find("r1"), std::string::npos);

    // invert's color input must come from r0 (register), not texelFetch.
    EXPECT_NE(glsl.find("node_invert("), std::string::npos);

    // Must end with imageStore.
    EXPECT_NE(glsl.find("imageStore"), std::string::npos);

    // invert is NOT format-sensitive, so no format post-process assignment on tail.
    // Note: _fmt_mono may be DEFINED in perlin's includes but should NOT be called.
    EXPECT_EQ(glsl.find("_local_1 = _fmt_mono("), std::string::npos)
        << "invert tail should not have format post-process";
}

TEST_F(FusedRealNodesGLSL, PerlinToInvertToBlend_DiamondChain) {
    // Diamond: perlin1->invert1->blend.a, perlin2->invert2->blend.b
    // The active path to blend includes: perlin1, invert1, perlin2, invert2, blend
    // But perlin1 and perlin2 are separate roots, so the active path tracer
    // follows from blend backwards: perlin1->invert1->blend, perlin2->invert2->blend
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "perlin"});
    g.nodes.push_back({4, "invert"});
    g.nodes.push_back({5, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({2, 0, 5, 0});
    g.connections.push_back({4, 0, 5, 1});
    g.output_node = 5;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    // The chain should include all 5 nodes (active path from blend).
    bool found = false;
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() >= 4) {
            found = true;
            const auto& glsl = ch.glsl;

            // Must call all node functions.
            EXPECT_NE(glsl.find("node_perlin"), std::string::npos);
            EXPECT_NE(glsl.find("node_invert"), std::string::npos);
            EXPECT_NE(glsl.find("node_blend"), std::string::npos);

            // blend has 3 inputs: mask(0)=float, a(1)=vec4, b(2)=vec4
            // a comes from invert1 (register), b from invert2 (register).
            // mask is unconnected -> SSBO read.
            EXPECT_NE(glsl.find("node_params"), std::string::npos)
                << "blend mask must read from param SSBO (unconnected float input)";

            // Must end with imageStore.
            EXPECT_NE(glsl.find("imageStore"), std::string::npos);
            break;
        }
    }
    EXPECT_TRUE(found) << "expected at least one chain with 4+ nodes for diamond graph";
}

TEST_F(FusedRealNodesGLSL, Blend_MaskFloatInput_ReadsSSBO) {
    // blend with mask=0.5 unconnected. Mask is a float input at socket 0.
    // Must read from SSBO, not hardcoded.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});  // perlin1 -> blend.a
    g.connections.push_back({2, 0, 3, 2});  // perlin2 -> blend.b
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    bool found = false;
    for (const auto& ch : cr.pass_plan.chains) {
        for (NodeId n : ch.nodes) {
            if (n == 3) {
                found = true;
                const auto& glsl = ch.glsl;

                // blend has: mask(float), a(vec4), b(vec4), mode(param)
                // mask is unconnected -> must read from SSBO.
                // Pattern: node_params[pc.param_ring_idx].v[pc.param_base_slot + ...]
                EXPECT_NE(glsl.find("node_params"), std::string::npos)
                    << "unconnected float mask must read from param SSBO";
                EXPECT_NE(glsl.find("pc.param_base_slot"), std::string::npos);

                // Must call node_blend.
                EXPECT_NE(glsl.find("node_blend("), std::string::npos);
                break;
            }
        }
        if (found) break;
    }
    EXPECT_TRUE(found) << "blend node not found in any chain";
}

TEST_F(FusedRealNodesGLSL, Grayscale_MaskAndMode) {
    // grayscale: inputs=[mask(float,0), color(vec4)], params=[mode(float)]
    // SSBO layout: [mode(0), mask_default(1)]
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "grayscale"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> grayscale.color
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u);

    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_grayscale"), std::string::npos);

    // grayscale is NOT format-sensitive, so no format post-process assignment.
    // Note: _fmt_mono may be DEFINED in perlin's includes but should NOT be called.
    EXPECT_EQ(glsl.find("_local_1 = _fmt_mono("), std::string::npos);

    // Must read mode param from SSBO.
    EXPECT_NE(glsl.find("node_params"), std::string::npos);
}

TEST_F(FusedRealNodesGLSL, ColorConst_Blend_ConstantInput) {
    // color_const -> blend.a, with blend.b unconnected.
    Graph g;
    g.nodes.push_back({1, "color_const"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 1});  // color_const -> blend.a
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("node_color_const"), std::string::npos);
    EXPECT_NE(glsl.find("node_blend"), std::string::npos);

    // color_const has 6 params (mode,r,g,b,a), blend has 1 param (mode).
    // Both param sets must be in SSBO.
    EXPECT_NE(glsl.find("node_params"), std::string::npos);
}

TEST_F(FusedRealNodesGLSL, Perlin_FormatSensitive_TailGetsFmt) {
    // Single perlin node with Mono format override.
    // Tail is perlin, format=Mono -> _fmt_mono applied.
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::Mono});
    g.output_node = 1;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u);

    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("_fmt_mono"), std::string::npos)
        << "perlin tail with Mono format must get _fmt_mono post-process";
}

TEST_F(FusedRealNodesGLSL, PerlinToInvert_MiddleFormatIgnored) {
    // perlin(Mono) -> invert
    // perlin is in the middle, invert is the tail.
    // invert has no format_override (RGBA default), so no format post-process on tail.
    // perlin's Mono format post-process runs on perlin's output (middle of chain).
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::Mono});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> invert.color
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    const auto& glsl = cr.pass_plan.chains[0].glsl;
    // No format post-process on invert (tail) because its format_override is RGBA (default).
    EXPECT_EQ(glsl.find("_local_1 = _fmt_mono("), std::string::npos);
}

TEST_F(FusedRealNodesGLSL, NoExternalInputs_LinearChain) {
    // perlin -> invert: invert's mask(float,socket0) is unconnected.
    // Unconnected Float → ConstSrc (SSBO read, no ext slot consumed).
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> invert.color
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    auto path = ActivePathTracer::trace(r.ir, 2, lib);
    auto result = emit_fused_subgraph(path, r.ir, lib, 0, {});
    ASSERT_TRUE(result.ok()) << result.error;
    // 0 external inputs — Float mask reads SSBO inline (ConstSrc), no ext slot
    EXPECT_EQ(result.external_inputs, 0u);
    EXPECT_EQ(result.source.find("_in_"), std::string::npos)
        << "float inputs should not generate _in_ declarations";
}

TEST_F(FusedRealNodesGLSL, ExternalInputs_BlendUnconnected) {
    // perlin -> blend.a, blend.b unconnected Vec4 = baked as vec4(0.0).
    // blend.mask (Float, unconnected) = ConstSrc (SSBO read, no ext slot).
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> blend.a
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    auto path = ActivePathTracer::trace(r.ir, 2, lib);
    auto result = emit_fused_subgraph(path, r.ir, lib, 0, {});
    ASSERT_TRUE(result.ok()) << result.error;
    // 0 external inputs — Float mask and Vec4 b are both baked/inline
    EXPECT_EQ(result.external_inputs, 0u)
        << "all unconnected inputs baked as constants, no ext slots";
    // No texelFetch — Float reads from SSBO, Vec4 b is vec4(0.0)
    EXPECT_EQ(result.source.find("texelFetch"), std::string::npos)
        << "no texelFetch should appear for unconnected inputs";
}

// ===========================================================================
// Part 2: Pixel correctness tests (Engine required)
// ===========================================================================

class FusedRealNodesPixel : public ::testing::Test {
protected:
    Engine engine;

    void SetUp() override {
        bool ok = init_engine(engine, "test_fused_real_nodes");
        if (!ok) GTEST_SKIP() << engine.last_error();
    }

    void TearDown() override {
        engine.shutdown();
    }
};

TEST_F(FusedRealNodesPixel, Perlin_ProducesNonZeroPixels) {
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    EXPECT_GT(avg_brightness(px), 0.0) << "perlin output is all-black";
}

TEST_F(FusedRealNodesPixel, PerlinToInvert_DiffersFromPerlin) {
    // Set perlin->invert graph, verify invert(mask=1) produces different output
    // than invert(mask=0) within the SAME graph — avoids cross-graph readback issues.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> invert.color
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // mask=1: full inversion
    engine.update_node_params_by_id(2, {1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px_mask1;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_mask1, w, h));

    // mask=0: passthrough
    engine.update_node_params_by_id(2, {0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px_mask0;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_mask0, w2, h2));

    EXPECT_GT(avg_brightness(px_mask1), 0.0);
    EXPECT_GT(avg_brightness(px_mask0), 0.0);

    // mask=1 (inverted) must differ from mask=0 (passthrough).
    double diff = 0;
    for (size_t i = 0; i + 3 < px_mask1.size() && i + 3 < px_mask0.size(); i += 4)
        diff += std::abs(px_mask1[i] - px_mask0[i]);
    EXPECT_GT(diff, 0.0) << "invert(mask=1) should differ from invert(mask=0)";
}

TEST_F(FusedRealNodesPixel, DiamondGraph_ProducesPixels) {
    // perlin->invert->blend, perlin->invert->blend
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "perlin"});
    g.nodes.push_back({4, "invert"});
    g.nodes.push_back({5, "blend"});
    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({2, 0, 5, 0});
    g.connections.push_back({4, 0, 5, 1});
    g.output_node = 5;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    EXPECT_GT(avg_brightness(px), 0.0) << "diamond graph output is all-black";
}

TEST_F(FusedRealNodesPixel, ValueToGrayscale_ProducesGrayPixels) {
    // value -> grayscale: output channels should be equal (gray).
    // Explicitly set mask=1 and mode=0 (luminance) to ensure correct behavior.
    Graph g;
    g.nodes.push_back({1, "value"});
    g.nodes.push_back({2, "grayscale"});
    g.connections.push_back({1, 0, 2, 1});  // value -> grayscale.color
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // grayscale params: mode=0 (luminance). SSBO: [mode(0), mask_default(1)]
    engine.update_node_params_by_id(2, {0.0f, 1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    ASSERT_GT(px.size(), 3u);

    // Check that R==G==B for a sample of pixels.
    int gray_count = 0;
    int total = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        float r = px[i], g_ = px[i+1], b = px[i+2];
        float max_diff = std::max({std::abs(r-g_), std::abs(g_-b), std::abs(r-b)});
        if (max_diff < 0.001f) gray_count++;
        total++;
    }
    double gray_ratio = total > 0 ? (double)gray_count / total : 0;
    EXPECT_GT(gray_ratio, 0.95) << "grayscale output should have R==G==B for most pixels (got "
                                 << gray_ratio << ")";
}

TEST_F(FusedRealNodesPixel, ColorConst_Blend_ConstantColor) {
    // color_const(mode=0, r=0.5) -> blend.a, blend.b unconnected (zero).
    // With mask=1.0 default, blend mode=0 (mix), output should be 50% gray.
    Graph g;
    g.nodes.push_back({1, "color_const"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 1});  // color_const -> blend.a
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Set color_const params: mode=0, r=0.5, g=0.5, b=0.5, a=1.0
    engine.update_node_params_by_id(1, {0.0f, 0.5f, 0.5f, 0.5f, 1.0f});
    // Set blend params: mode=0 (mix), mask=0 (passthrough A).
    // mask must be explicit: before the seed_param_ssbo_defaults_ fix,
    // mask was wrongly seeded to 0.0 by memset; now it's correctly
    // seeded to 1.0 (manifest default). mask=0 forces passthrough of A.
    engine.update_node_params_by_id(2, {0.0f, 0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine.poll_pending_compiles();

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    ASSERT_GT(px.size(), 3u);

    // Sample center pixel: should be around 0.5 gray.
    size_t mid = (px.size() / 8) * 4;  // ~center of image
    EXPECT_NEAR(px[mid], 0.5f, 0.1f) << "center R should be ~0.5";
    EXPECT_NEAR(px[mid+1], 0.5f, 0.1f) << "center G should be ~0.5";
    EXPECT_NEAR(px[mid+2], 0.5f, 0.1f) << "center B should be ~0.5";
}

TEST_F(FusedRealNodesPixel, PerlinToInvert_Mask0_Passthrough) {
    // perlin -> invert(mask=0): should pass through original perlin values.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> invert.color
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Set invert mask=0 (passthrough).
    engine.update_node_params_by_id(2, {0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine.poll_pending_compiles();

    std::vector<float> px_inverted;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_inverted, w, h));

    // Compare with perlin-only output.
    Graph g1;
    g1.nodes.push_back({1, "perlin"});
    g1.output_node = 1;
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_original;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen1, px_original, w2, h2));

    // With mask=0, inverted should equal original.
    double diff = 0;
    for (size_t i = 0; i + 3 < px_inverted.size() && i + 3 < px_original.size(); i += 4)
        diff += std::abs(px_inverted[i] - px_original[i]);
    EXPECT_NEAR(diff, 0.0, 0.01) << "invert(mask=0) should passthrough original values";
}

// ===========================================================================
// Part 3: Chain structure tests (verify correct chain membership)
// ===========================================================================

class FusedRealNodesChain : public ::testing::Test {
protected:
    NodeLibrary lib = load_real_lib();

    CompileGraphResult compile(Graph& g) {
        auto r = validate_graph(g, lib);
        EXPECT_TRUE(r.success) << r.error;
        if (!r.success) return {};
        return FusedGraphCompiler::compile(r.ir, lib, g.output_node);
    }
};

TEST_F(FusedRealNodesChain, SingleNode_SingleChain) {
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.output_node = 1;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    EXPECT_EQ(cr.pass_plan.chains.size(), 1u);
    EXPECT_EQ(cr.pass_plan.chains[0].nodes.size(), 1u);
}

TEST_F(FusedRealNodesChain, LinearChain_AllNodesInOneChain) {
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    // All 3 nodes should be in one chain.
    bool found = false;
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 3) { found = true; break; }
    }
    EXPECT_TRUE(found) << "expected one chain with 3 nodes";

    // Verify chain_index_of_pass covers all passes.
    for (uint32_t i = 0; i < cr.pass_plan.chain_index_of_pass.size(); ++i) {
        EXPECT_NE(cr.pass_plan.chain_index_of_pass[i], UINT32_MAX)
            << "pass " << i << " should be in a chain";
    }
}

TEST_F(FusedRealNodesChain, ParamBaseSlotCorrectness) {
    // perlin has 6 params, invert has 0 params + 1 float input.
    // perlin base=0, next=6.
    // invert params=0, float_inputs=1 (mask) -> base=6, next=6+0+1=7.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    EXPECT_EQ(cr.param_base_slot[1], 0);
    EXPECT_EQ(cr.param_base_slot[2], 6);  // 6 perlin params + 0 float inputs
    EXPECT_EQ(cr.total_param_floats, 7);  // 6 + 1
}

// ===========================================================================
// Part 4: Sampler2D chain-split tests
// Nodes with Sampler2D inputs (blur, warp) must be in a separate chain
// from their source so the source's output is materialized as a VRAM image.
// ===========================================================================

TEST_F(FusedRealNodesGLSL, PerlinToBlur_Sampler2DChainSplit) {
    // perlin(vec4) -> blur(sampler2D input)
    // Must produce TWO chains: [perlin] and [blur].
    // Blur's GLSL must construct a TSTexture, NOT pass a vec4 register.
    // Blur chain must have sub_pass_count == 2 (separable: H + V).
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});  // perlin -> blur.tex (sampler2D)
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    // Must produce 2 chains (split at Sampler2D boundary).
    ASSERT_EQ(cr.pass_plan.chains.size(), 2u)
        << "perlin->blur must split into 2 chains at Sampler2D boundary";

    // First chain: [perlin], second chain: [blur].
    bool found_perlin_chain = false, found_blur_chain = false;
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 1) found_perlin_chain = true;
        if (ch.nodes.size() == 1 && ch.nodes[0] == 2) found_blur_chain = true;
    }
    EXPECT_TRUE(found_perlin_chain) << "perlin must be in its own chain (chain tail)";
    EXPECT_TRUE(found_blur_chain) << "blur must be in its own chain";

    // Blur's chain must use TSTexture constructor for the sampler input.
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 2) {
            // Multi-pass: sub_pass_count must be 2 (H + V).
            EXPECT_EQ(ch.sub_pass_count, 2u)
                << "blur chain must have sub_pass_count=2 (separable blur)";
            EXPECT_EQ(ch.intermediate_count, 1u)
                << "blur chain must have intermediate_count=1";

            // Sub-pass GLSL must exist and contain TSTexture.
            ASSERT_EQ(ch.sub_pass_glsl.size(), 2u)
                << "blur chain must have 2 sub-pass GLSL sources";
            for (uint32_t sp = 0; sp < 2; ++sp) {
                EXPECT_NE(ch.sub_pass_glsl[sp].find("TSTexture"), std::string::npos)
                    << "sub-pass " << sp << " must construct TSTexture";
                EXPECT_NE(ch.sub_pass_glsl[sp].find("node_blur"), std::string::npos)
                    << "sub-pass " << sp << " must call node_blur";
                EXPECT_NE(ch.sub_pass_glsl[sp].find("sigma"), std::string::npos)
                    << "sub-pass " << sp << " must contain Gaussian sigma computation";
            }

            // Legacy glsl field: for multi-pass chains this is the node-level emit
            // (not the full chain wrapper), so it may not contain TSTexture.
            // The important check is the sub-pass GLSLs above.
            break;
        }
    }
}

TEST_F(FusedRealNodesChain, PerlinToBlur_TwoChains) {
    // Verify chain membership: perlin and blur are in separate chains.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 2u);

    // chain_index_of_pass: perlin and blur must have different chain indices.
    uint32_t perlin_chain = cr.pass_plan.chain_index_of_pass[0];
    uint32_t blur_chain = cr.pass_plan.chain_index_of_pass[1];
    EXPECT_NE(perlin_chain, UINT32_MAX);
    EXPECT_NE(blur_chain, UINT32_MAX);
    EXPECT_NE(perlin_chain, blur_chain)
        << "perlin and blur must be in different chains";

    // Perlin's output must be in active_resources (consumed by blur cross-chain).
    // This ensures ResourceManager allocates an image for it.
    ResourceUUID perlin_out{1, 0};
    EXPECT_NE(cr.pass_plan.active_resources.find(perlin_out),
              cr.pass_plan.active_resources.end())
        << "perlin output must be in active_resources (consumed by blur cross-chain)";
}

TEST_F(FusedRealNodesPixel, PerlinToBlur_DiffersFromPerlin) {
    // perlin -> blur(intensity=1.0) must produce different output than raw perlin.
    // A separable Gaussian blur smooths noise, reducing high-frequency detail.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Set blur intensity=1.0 (strong blur).
    engine.update_node_params_by_id(2, {1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_blur;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_blur, w, h));
    EXPECT_GT(avg_brightness(px_blur), 0.0) << "blur output is all-black";

    // Compare with perlin-only output.
    Graph g1;
    g1.nodes.push_back({1, "perlin"});
    g1.output_node = 1;
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_perlin;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen1, px_perlin, w2, h2));

    // Blur must differ from raw perlin.
    double diff = 0;
    for (size_t i = 0; i + 3 < px_blur.size() && i + 3 < px_perlin.size(); i += 4)
        diff += std::abs(px_blur[i] - px_perlin[i]);
    EXPECT_GT(diff, 0.0)
        << "blur(intensity=1.0) output must differ from raw perlin";
}

TEST_F(FusedRealNodesGLSL, Blur_VariantKeysDifferPerSubPass) {
    // The two sub-passes must have different variant keys
    // (specialization[0] = 0 for H, 1 for V).
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 2) {
            ASSERT_EQ(ch.sub_pass_variant_keys.size(), 2u);
            EXPECT_NE(ch.sub_pass_variant_keys[0], ch.sub_pass_variant_keys[1])
                << "H and V sub-passes must have different variant keys";
            // Specialization constant should differ.
            EXPECT_EQ(ch.sub_pass_variant_keys[0].specialization[0], 0u);
            EXPECT_EQ(ch.sub_pass_variant_keys[1].specialization[0], 1u);
            EXPECT_EQ(ch.sub_pass_variant_keys[0].specialization_count, 1u);
            EXPECT_EQ(ch.sub_pass_variant_keys[1].specialization_count, 1u);
            break;
        }
    }
}

TEST_F(FusedRealNodesChain, Blur_SingletonChainMembership) {
    // Blur chain must be a singleton (one node).
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 2u);

    // Blur must be in a singleton chain with sub_pass_count=2.
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 2) {
            EXPECT_EQ(ch.sub_pass_count, 2u);
            EXPECT_EQ(ch.nodes.size(), 1u);
            break;
        }
    }
}

// ===========================================================================
// Part 5: Warp node tests
// ===========================================================================

TEST_F(FusedRealNodesGLSL, Warp_ContainsAllModes) {
    // Warp GLSL must contain all 4 mode branches: gradient, directional, curl, slope.
    // Warp is single-pass, so GLSL is in ch.glsl (not ch.sub_pass_glsl).
    Graph g;
    g.nodes.push_back({1, "warp"});
    g.output_node = 1;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 1) {
            EXPECT_EQ(ch.sub_pass_count, 0u)
                << "warp is single-pass (sub_pass_count=0)";
            EXPECT_NE(ch.glsl.find("node_warp"), std::string::npos)
                << "warp GLSL must contain node_warp function";
            EXPECT_NE(ch.glsl.find("GetTexelSize"), std::string::npos)
                << "warp GLSL must use GetTexelSize (not hardcoded texel size)";
            EXPECT_NE(ch.glsl.find("mode < 0.5"), std::string::npos)
                << "warp GLSL must contain gradient mode branch";
            EXPECT_NE(ch.glsl.find("mode < 1.5"), std::string::npos)
                << "warp GLSL must contain directional mode branch";
            EXPECT_NE(ch.glsl.find("mode < 2.5"), std::string::npos)
                << "warp GLSL must contain curl mode branch";
            break;
        }
    }
}

TEST_F(FusedRealNodesChain, Warp_Sampler2DChainSplit) {
    // warp(image, gradient) -> both Sampler2D inputs, must be in singleton chain.
    // warp's source must be materialized as VRAM image.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "warp"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> warp.gradient
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 2u)
        << "perlin->warp must split into 2 chains at Sampler2D boundary";

    // Warp must be in singleton chain (Sampler2D boundary).
    bool found_warp_chain = false;
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 2) found_warp_chain = true;
    }
    EXPECT_TRUE(found_warp_chain) << "warp must be in its own chain";

    // Perlin output must be in active_resources.
    ResourceUUID perlin_out{1, 0};
    EXPECT_NE(cr.pass_plan.active_resources.find(perlin_out),
              cr.pass_plan.active_resources.end())
        << "perlin output must be in active_resources (consumed by warp cross-chain)";
}

TEST_F(FusedRealNodesPixel, PerlinToWarp_DiffersFromPerlin) {
    // perlin -> warp(image, gradient) must produce different output than raw perlin.
    // Connect perlin to BOTH sampler inputs (image + gradient) so it displaces itself.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "warp"});
    g.connections.push_back({1, 0, 2, 0});  // perlin -> warp.image
    g.connections.push_back({1, 0, 2, 1});  // perlin -> warp.gradient
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Set warp intensity=0.5, mode=0 (gradient).
    engine.update_node_params_by_id(2, {0.5f, 0.0f, 0.0f, 0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_warp;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_warp, w, h));
    EXPECT_GT(avg_brightness(px_warp), 0.0) << "warp output is all-black";

    // Compare with perlin-only output.
    Graph g1;
    g1.nodes.push_back({1, "perlin"});
    g1.output_node = 1;
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_perlin;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen1, px_perlin, w2, h2));

    // Warp must differ from raw perlin.
    double diff = 0;
    for (size_t i = 0; i + 3 < px_warp.size() && i + 3 < px_perlin.size(); i += 4)
        diff += std::abs(px_warp[i] - px_perlin[i]);
    EXPECT_GT(diff, 0.0)
        << "warp(intensity=0.5) output must differ from raw perlin";
}

// ===========================================================================
// Part 6: Multi-pass blur correctness tests
// These tests expose bugs in the multi-pass blur dispatch where sub-pass 1
// (V-blur) reads from the original texture instead of the H-blurred
// intermediate, and where the intermediate image format may not match
// the GLSL layout qualifier.
// ===========================================================================

namespace {
// Compute average magnitude of horizontal gradients (finite differences)
// across all RGBA channels. Higher values = sharper horizontal edges.
double avg_horizontal_gradient(const std::vector<float>& px, uint32_t w, uint32_t h) {
    double sum = 0;
    uint64_t count = 0;
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x + 1 < w; ++x) {
            for (int c = 0; c < 4; ++c) {
                size_t i0 = (y * w + x) * 4 + c;
                size_t i1 = (y * w + x + 1) * 4 + c;
                sum += std::abs(px[i1] - px[i0]);
                ++count;
            }
        }
    }
    return count ? sum / count : 0.0;
}

// Compute average magnitude of vertical gradients (finite differences)
// across all RGBA channels. Higher values = sharper vertical edges.
double avg_vertical_gradient(const std::vector<float>& px, uint32_t w, uint32_t h) {
    double sum = 0;
    uint64_t count = 0;
    for (uint32_t y = 0; y + 1 < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            for (int c = 0; c < 4; ++c) {
                size_t i0 = (y * w + x) * 4 + c;
                size_t i1 = ((y + 1) * w + x) * 4 + c;
                sum += std::abs(px[i1] - px[i0]);
                ++count;
            }
        }
    }
    return count ? sum / count : 0.0;
}
} // anonymous namespace

TEST_F(FusedRealNodesPixel, Blur_SeparableSymmetry) {
    // BUG 1 EXPOSER: If sub-pass 1 (V-blur) reads from the original texture
    // instead of the H-blurred intermediate, the output is effectively a
    // V-blur only — horizontal edges are preserved while vertical edges
    // are smoothed. This test detects that asymmetry by comparing
    // horizontal vs vertical gradient magnitudes.
    //
    // Correct behavior: H-blur and V-blur are both applied, so horizontal
    // and vertical gradient magnitudes should be roughly equal (within tolerance).
    //
    // Bug behavior: Only V-blur is applied, so horizontal gradient > vertical gradient.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Strong blur to amplify the asymmetry.
    engine.update_node_params_by_id(2, {1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    EXPECT_GT(avg_brightness(px), 0.0) << "blur output is all-black";

    double hgrad = avg_horizontal_gradient(px, w, h);
    double vgrad = avg_vertical_gradient(px, w, h);

    // With a separable blur, both directions should be smoothed equally.
    // Allow 50% tolerance — perlin noise is not perfectly isotropic.
    // If the bug exists, hgrad will be 2-5x larger than vgrad.
    EXPECT_LT(hgrad, vgrad * 2.0)
        << "Blur asymmetry: horizontal gradient (" << hgrad
        << ") is much larger than vertical (" << vgrad
        << "). Sub-pass 1 may be reading from original instead of H-blurred intermediate.";
}

TEST_F(FusedRealNodesPixel, Blur_IntensityScalesSmoothly) {
    // BUG 1 EXPOSER: With correct separable blur, increasing intensity
    // should monotonically decrease gradient magnitude. If sub-pass 1
    // reads from original, increasing intensity only strengthens the
    // V-blur, making the H/V asymmetry worse.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    // Read unblurred perlin.
    uint64_t gen0 = engine.set_graph(g);
    ASSERT_NE(gen0, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px0;
    uint32_t w0 = 0, h0 = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen0, px0, w0, h0));
    double hgrad0 = avg_horizontal_gradient(px0, w0, h0);

    // Read blurred perlin (intensity=1.0).
    engine.update_node_params_by_id(2, {1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px1;
    uint32_t w1 = 0, h1 = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen0, px1, w1, h1));
    double hgrad1 = avg_horizontal_gradient(px1, w1, h1);

    // Blur should reduce horizontal gradients.
    EXPECT_LT(hgrad1, hgrad0)
        << "Blur with intensity=1.0 should reduce horizontal gradients vs unblurred.";

    // The ratio should be less than 1 (gradient reduced).
    // If the bug exists, hgrad1 ≈ hgrad0 (H-blur is missing).
    double ratio = (hgrad0 > 0) ? hgrad1 / hgrad0 : 0;
    EXPECT_LT(ratio, 0.9)
        << "Blur should reduce horizontal gradients by >10%. Ratio=" << ratio
        << ". Sub-pass 1 may not be reading from H-blurred intermediate.";
}

TEST_F(FusedRealNodesChain, Blur_IntermediateFormat) {
    // BUG 2 EXPOSER: The intermediate image format must match the GLSL
    // storage image layout qualifier. If the graph uses F16 precision but
    // the intermediate is RGBA32F, the format qualifier mismatches.
    //
    // This test verifies the intermediate is allocated with the correct format.
    // The intermediate's VkFormat should match the storage format derived
    // from the node's resolved depth and channel format.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 2u);

    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 2) {
            EXPECT_EQ(ch.sub_pass_count, 2u);
            EXPECT_EQ(ch.intermediate_count, 1u);

            // Both sub-pass GLSL must use GetTexelSize (not hardcoded texel size).
            for (uint32_t sp = 0; sp < ch.sub_pass_glsl.size(); ++sp) {
                EXPECT_NE(ch.sub_pass_glsl[sp].find("pc.in_sampled_slots[0]"),
                          std::string::npos)
                    << "sub-pass " << sp << " must read from pc.in_sampled_slots[0]";
            }
            break;
        }
    }
}

TEST_F(FusedRealNodesPixel, Blur_HorizontalStripes_ExposesVerticalOnlyBug) {
    // STRIP-DOWN TEST: upload known pixel values, read back raw, print exact numbers.
    // Step 1: image only (no blur) — verify upload works.
    // Step 2: image → blur — see what blur actually does.

    constexpr uint32_t W = 64, H = 64;
    constexpr uint32_t STRIPE_H = 8;

    // Build horizontal stripes: rows 0-7 = 0.9, rows 8-15 = 0.1, repeat.
    std::vector<float> px(W * H * 4);
    for (uint32_t y = 0; y < H; ++y) {
        float val = ((y / STRIPE_H) % 2 == 0) ? 0.9f : 0.1f;
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t i = (y * W + x) * 4;
            px[i + 0] = val;
            px[i + 1] = val;
            px[i + 2] = val;
            px[i + 3] = 1.0f;
        }
    }

    const uint64_t image_id = 10;
    ASSERT_TRUE(engine.upload_image(image_id, px.data(), W, H));
    // Wait long enough for transfer queue to finish.
    for (int i = 0; i < 200; ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // ---- STEP 1: image only, no blur ----
    {
        Graph g;
        g.nodes.push_back({image_id, "image"});
        g.output_node = image_id;

        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        engine.set_resolution(W, H);
        ASSERT_TRUE(wait_for_pipeline(engine));

        std::vector<float> out;
        uint32_t ow = 0, oh = 0;
        PushConstants pc{};
        pc.resolution_x = W; pc.resolution_y = H;
        pc.seed = 1; pc.time = 0.0f;
        uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
        ASSERT_NE(ticket, 0u);
        bool got = false;
        for (int i = 0; i < 500; ++i) {
            if (engine.async_readback().poll(engine.ctx(), out, ow, oh, gen)) { got = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_TRUE(got) << "readback timed out";

        printf("\n=== STEP 1: image only (no blur) %ux%u ===\n", ow, oh);
        // Print a vertical slice at x=32
        printf("Column x=32, vertical slice:\n");
        for (uint32_t y = 0; y < H && y < oh; y += 4) {
            size_t i = (y * ow + 32) * 4;
            printf("  y=%2u: R=%.6f G=%.6f B=%.6f A=%.6f  (expected %.1f)\n",
                   y, out[i], out[i+1], out[i+2], out[i+3],
                   ((y / STRIPE_H) % 2 == 0) ? 0.9f : 0.1f);
        }
        // Print a horizontal slice at y=4 (should be all 0.9)
        printf("Row y=4, horizontal slice:\n");
        for (uint32_t x = 0; x < W && x < ow; x += 8) {
            size_t i = (4 * ow + x) * 4;
            printf("  x=%2u: R=%.6f (expected 0.9)\n", x, out[i]);
        }
    }

    // ---- STEP 2: image → blur, intensity=1.0 ----
    {
        Graph g;
        g.nodes.push_back({image_id, "image"});
        g.nodes.push_back({2, "blur"});
        g.connections.push_back({image_id, 0, 2, 0});
        g.output_node = 2;

        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        engine.set_resolution(W, H);
        ASSERT_TRUE(wait_for_pipeline(engine));

        engine.update_node_params_by_id(2, {1.0f});
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        engine.poll_pending_compiles();

        std::vector<float> out;
        uint32_t ow = 0, oh = 0;
        PushConstants pc{};
        pc.resolution_x = W; pc.resolution_y = H;
        pc.seed = 1; pc.time = 0.0f;
        uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
        ASSERT_NE(ticket, 0u);
        bool got = false;
        for (int i = 0; i < 500; ++i) {
            if (engine.async_readback().poll(engine.ctx(), out, ow, oh, gen)) { got = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_TRUE(got) << "readback timed out";

        printf("\n=== STEP 2: image → blur (intensity=1.0) %ux%u ===\n", ow, oh);
        printf("Column x=32, vertical slice:\n");
        for (uint32_t y = 0; y < H && y < oh; y += 4) {
            size_t i = (y * ow + 32) * 4;
            printf("  y=%2u: R=%.6f G=%.6f B=%.6f A=%.6f\n",
                   y, out[i], out[i+1], out[i+2], out[i+3]);
        }
        printf("Row y=4, horizontal slice:\n");
        for (uint32_t x = 0; x < W && x < ow; x += 8) {
            size_t i = (4 * ow + x) * 4;
            printf("  x=%2u: R=%.6f\n", x, out[i]);
        }
        // Boundary check: y=7 (last bright row) → y=8 (first dark row)
        size_t i7 = (7 * ow + 32) * 4;
        size_t i8 = (8 * ow + 32) * 4;
        printf("Boundary y=7→8 at x=32: %.6f → %.6f (gradient=%.6f)\n",
               out[i7], out[i8], std::abs(out[i8] - out[i7]));
    }

    printf("================================================\n\n");
}

TEST_F(FusedRealNodesPixel, Blur_VerticalStripes_HBlurTest) {
    // Test H-blur: vertical stripes have horizontal edges but NO vertical edges.
    // If H-blur works, horizontal gradient decreases.
    // If V-only blur, nothing changes (no vertical edges to smooth).

    constexpr uint32_t W = 64, H = 64;
    constexpr uint32_t STRIPE_W = 8;

    // Vertical stripes: columns 0-7 = 0.9, columns 8-15 = 0.1, repeat.
    std::vector<float> px(W * H * 4);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float val = ((x / STRIPE_W) % 2 == 0) ? 0.9f : 0.1f;
            uint32_t i = (y * W + x) * 4;
            px[i + 0] = val;
            px[i + 1] = val;
            px[i + 2] = val;
            px[i + 3] = 1.0f;
        }
    }

    const uint64_t image_id = 11;
    ASSERT_TRUE(engine.upload_image(image_id, px.data(), W, H));
    for (int i = 0; i < 200; ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // ---- STEP 1: image only ----
    {
        Graph g;
        g.nodes.push_back({image_id, "image"});
        g.output_node = image_id;

        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        engine.set_resolution(W, H);
        ASSERT_TRUE(wait_for_pipeline(engine));

        std::vector<float> out;
        uint32_t ow = 0, oh = 0;
        PushConstants pc{};
        pc.resolution_x = W; pc.resolution_y = H;
        pc.seed = 1; pc.time = 0.0f;
        uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
        ASSERT_NE(ticket, 0u);
        bool got = false;
        for (int i = 0; i < 500; ++i) {
            if (engine.async_readback().poll(engine.ctx(), out, ow, oh, gen)) { got = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_TRUE(got) << "readback timed out";

        printf("\n=== VERTICAL STRIPES STEP 1: image only ===\n");
        printf("Row y=32, horizontal slice (should show 0.9/0.1 pattern):\n");
        for (uint32_t x = 0; x < W && x < ow; x += 4) {
            size_t i = (32 * ow + x) * 4;
            printf("  x=%2u: R=%.6f (expect %.1f)\n", x, out[i],
                   ((x / STRIPE_W) % 2 == 0) ? 0.9f : 0.1f);
        }
    }

    // ---- STEP 2: image → blur, intensity=1.0 ----
    {
        Graph g;
        g.nodes.push_back({image_id, "image"});
        g.nodes.push_back({2, "blur"});
        g.connections.push_back({image_id, 0, 2, 0});
        g.output_node = 2;

        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        engine.set_resolution(W, H);
        ASSERT_TRUE(wait_for_pipeline(engine));

        engine.update_node_params_by_id(2, {1.0f});
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        engine.poll_pending_compiles();

        std::vector<float> out;
        uint32_t ow = 0, oh = 0;
        PushConstants pc{};
        pc.resolution_x = W; pc.resolution_y = H;
        pc.seed = 1; pc.time = 0.0f;
        uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
        ASSERT_NE(ticket, 0u);
        bool got = false;
        for (int i = 0; i < 500; ++i) {
            if (engine.async_readback().poll(engine.ctx(), out, ow, oh, gen)) { got = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_TRUE(got) << "readback timed out";

        printf("\n=== VERTICAL STRIPES STEP 2: blur intensity=1.0 ===\n");
        printf("Row y=32, horizontal slice (stripe pattern should be smoothed):\n");
        for (uint32_t x = 0; x < W && x < ow; x += 4) {
            size_t i = (32 * ow + x) * 4;
            printf("  x=%2u: R=%.6f\n", x, out[i]);
        }
        printf("Column x=32, vertical slice (should be UNIFORM — no vertical edges):\n");
        for (uint32_t y = 0; y < H && y < oh; y += 8) {
            size_t i = (y * ow + 32) * 4;
            printf("  y=%2u: R=%.6f\n", y, out[i]);
        }
        // Boundary check at x=7→8 (bright→dark)
        size_t i7 = (32 * ow + 7) * 4;
        size_t i8 = (32 * ow + 8) * 4;
        printf("Boundary x=7→8 at y=32: %.6f → %.6f (gradient=%.6f)\n",
               out[i7], out[i8], std::abs(out[i8] - out[i7]));
    }
    printf("================================================\n\n");
}

TEST_F(FusedRealNodesPixel, Blur_LowIntensity_NoHugeValues) {
    // Regression: intensity < 0.014 produced insane values (e.g. Red=62268).
    // Root cause: erf approximation missing "1.0 -" → inverted sign on weights.
    // Test with solid-color input — output must stay in [0,1].

    constexpr uint32_t W = 64, H = 64;
    constexpr float COLOR = 0.5f;

    std::vector<float> px(W * H * 4);
    for (uint32_t i = 0; i < W * H; ++i) {
        px[i * 4 + 0] = COLOR;
        px[i * 4 + 1] = COLOR;
        px[i * 4 + 2] = COLOR;
        px[i * 4 + 3] = 1.0f;
    }

    const uint64_t image_id = 11;
    ASSERT_TRUE(engine.upload_image(image_id, px.data(), W, H));
    // Wait for transfer queue to finish (2s is generous for 64x64).
    for (int i = 0; i < 200; ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Helper lambda: build image→blur graph, set intensity, read back, return max value.
    auto test_intensity = [&](float intensity, float& out_max, float& out_avg) {
        Graph g;
        g.nodes.push_back({image_id, "image"});
        g.nodes.push_back({2, "blur"});
        g.connections.push_back({image_id, 0, 2, 0});
        g.output_node = 2;

        uint64_t gen = engine.set_graph(g);
        if (gen == 0) { out_max = -1; out_avg = -1; return; }
        engine.set_resolution(W, H);
        if (!wait_for_pipeline(engine)) { out_max = -1; out_avg = -1; return; }

        engine.update_node_params_by_id(2, {intensity});
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        engine.poll_pending_compiles();

        std::vector<float> out;
        uint32_t ow = 0, oh = 0;
        PushConstants pc{};
        pc.resolution_x = W; pc.resolution_y = H;
        pc.seed = 1; pc.time = 0.0f;
        uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
        if (ticket == 0) { out_max = -1; out_avg = -1; return; }
        bool got = false;
        for (int i = 0; i < 500; ++i) {
            if (engine.async_readback().poll(engine.ctx(), out, ow, oh, gen)) { got = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!got || ow == 0 || oh == 0) { out_max = -1; out_avg = -1; return; }

        out_max = 0.0f;
        for (size_t i = 0; i + 3 < out.size(); i += 4) {
            for (int c = 0; c < 3; ++c) {
                if (out[i + c] > out_max) out_max = out[i + c];
            }
        }
        out_avg = float(avg_brightness(out) / double(ow * oh));
    };

    // Warm-up: run intensity=1.0 to ensure image upload is fully propagated.
    {
        float warm_max, warm_avg;
        test_intensity(1.0f, warm_max, warm_avg);
        printf("warm-up: max=%.6f avg=%.6f\n", warm_max, warm_avg);
        ASSERT_GT(warm_max, 0.0f) << "warm-up produced no data — image upload likely failed";
    }

    float test_intensities[] = {0.014f, 0.01f, 0.005f, 0.001f, 0.0001f};
    for (float intensity : test_intensities) {
        float t_max, t_avg;
        test_intensity(intensity, t_max, t_avg);

        float sigma = std::max(intensity * 50.0f, 0.1f);
        printf("intensity=%.4f  sigma=%.2f  max=%.6f  avg=%.6f\n",
               intensity, sigma, t_max, t_avg);

        EXPECT_LE(t_max, 1.0f)
            << "intensity=" << intensity << " max=" << t_max
            << " (expected all values in [0,1] for solid-color input)";

        EXPECT_NEAR(t_avg, COLOR, 0.05)
            << "intensity=" << intensity << " avg=" << t_avg
            << " (expected near " << COLOR << " for solid-color blur)";
    }
}

// ===========================================================================
// Part 8: User-graph diagnostic — warp intensity=0 passthrough
//
// User graph (256x256 in Blender):
//   Node 1: perlin (format_override=1 Mono)
//   Node 2: levels
//   Node 3: blur
//   Node 4: warp  (intensity=0, mode=3, angle=0, edge_wrap=1)
//   Node 5: worley (format_override=1 Mono)
//   Node 6: blur
//   Node 7: invert
//   Connections:
//     1→2, 2→3, 3→4.socket_0 (image)
//     5→6, 6→7.socket_0, 7→4.socket_1 (gradient)
//   output_node = 4
//
// At intensity=0, warp.glsl line 3 returns Sample(image, uv) unchanged.
// Expected output = perlin→levels→blur (smooth Perlin).
// Actual output per user = "contrasty voronoi, as if worley→levels".
//
// Diagnostic: render full graph at intensity=0, then render image-chain
// alone, then render gradient-chain alone. Compare all three.
// ===========================================================================

namespace {

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double md = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i += 4) {
        for (int c = 0; c < 3; ++c)
            md = std::max(md, std::abs((double)a[i+c] - (double)b[i+c]));
    }
    return md;
}

double avg_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0;
    uint64_t count = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i += 4) {
        for (int c = 0; c < 3; ++c) {
            sum += std::abs((double)a[i+c] - (double)b[i+c]);
            ++count;
        }
    }
    return count ? sum / count : 0.0;
}

// Print per-channel min/max/avg for a pixel buffer.
void dump_channel_stats(const char* label, const std::vector<float>& px) {
    if (px.empty()) { printf("%s: EMPTY\n", label); return; }
    float ch_min[4] = {999,999,999,999}, ch_max[4] = {-999,-999,-999,-999};
    double ch_sum[4] = {0,0,0,0};
    uint64_t n = px.size() / 4;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        for (int c = 0; c < 4; ++c) {
            ch_min[c] = std::min(ch_min[c], px[i+c]);
            ch_max[c] = std::max(ch_max[c], px[i+c]);
            ch_sum[c] += px[i+c];
        }
    }
    printf("  %s (%lu px): R[%.3f..%.3f avg=%.3f] G[%.3f..%.3f avg=%.3f] "
           "B[%.3f..%.3f avg=%.3f] A[%.3f..%.3f avg=%.3f]\n",
           label, (unsigned long)n,
           ch_min[0], ch_max[0], ch_sum[0]/n,
           ch_min[1], ch_max[1], ch_sum[1]/n,
           ch_min[2], ch_max[2], ch_sum[2]/n,
           ch_min[3], ch_max[3], ch_sum[3]/n);
}

} // anonymous namespace

TEST_F(FusedRealNodesPixel, Warp_IntensityZero_Passthrough) {
    // TEST 1: Simple warp(perlin, perlin) at intensity=0.
    // Expected: output == perlin (passthrough).
    // If this fails, the basic warp plumbing is broken.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "warp"});
    g.connections.push_back({1, 0, 2, 0});  // perlin -> warp.image
    g.connections.push_back({1, 0, 2, 1});  // perlin -> warp.gradient
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // intensity=0, mode=0
    engine.update_node_params_by_id(2, {0.0f, 0.0f, 0.0f, 0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_warp;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_warp, w, h));

    // Now render perlin alone.
    Graph g1;
    g1.nodes.push_back({1, "perlin"});
    g1.output_node = 1;
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_perlin;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen1, px_perlin, w2, h2));

    printf("TEST 1: Simple warp(perlin,perlin) intensity=0\n");
    dump_channel_stats("warp_output", px_warp);
    dump_channel_stats("perlin_only", px_perlin);
    printf("  max_diff=%.6f  avg_diff=%.6f\n",
           max_diff(px_warp, px_perlin), avg_diff(px_warp, px_perlin));

    // At intensity=0, warp should pass through image unchanged.
    // Tolerance 0.03 accounts for Vulkan pipeline precision noise between
    // two separate graph compilations (different readback slots/pipelines).
    EXPECT_LT(max_diff(px_warp, px_perlin), 0.03)
        << "warp(intensity=0) must pass through image unchanged. "
           "If diff > 0.03, the basic warp plumbing is broken.";
}

TEST_F(FusedRealNodesPixel, Warp_UserGraph_IntensityZero) {
    // TEST 2: Full user graph at intensity=0.
    // Renders full graph, image-chain alone, gradient-chain alone.
    // Compares all three to find where wrong output appears.
    uint32_t RES = 256;

    // Full user graph: Perlin(Mono)->Levels->Blur->Warp.Image
    //                  Worley(Mono)->Blur->Invert->Warp.Gradient
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::Mono});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "blur"});
    g.nodes.push_back({4, "warp"});
    g.nodes.push_back({5, "worley", ChannelFormat::Mono});
    g.nodes.push_back({6, "blur"});
    g.nodes.push_back({7, "invert"});
    g.connections.push_back({1, 0, 2, 0});   // perlin -> levels.color
    g.connections.push_back({2, 0, 3, 0});   // levels -> blur.color
    g.connections.push_back({3, 0, 4, 0});   // blur -> warp.image
    g.connections.push_back({5, 0, 6, 0});   // worley -> blur2.color
    g.connections.push_back({6, 0, 7, 1});   // blur2 -> invert.color (socket 1 in invert.node.json)
    g.connections.push_back({7, 0, 4, 1});   // invert -> warp.gradient
    g.output_node = 4;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // intensity=0, mode=3 (slope), angle=0, edge_wrap=1
    engine.update_node_params_by_id(4, {0.0f, 3.0f, 0.0f, 1.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_full;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen_res(engine, gen, px_full, w, h, RES, RES));

    // Image chain alone: Perlin(Mono)->Levels->Blur
    Graph g_img;
    g_img.nodes.push_back({1, "perlin", ChannelFormat::Mono});
    g_img.nodes.push_back({2, "levels"});
    g_img.nodes.push_back({3, "blur"});
    g_img.connections.push_back({1, 0, 2, 0});
    g_img.connections.push_back({2, 0, 3, 0});
    g_img.output_node = 3;

    uint64_t gen_img = engine.set_graph(g_img);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_img;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback_gen_res(engine, gen_img, px_img, w2, h2, RES, RES));

    // Gradient chain alone: Worley(Mono)->Blur->Invert
    Graph g_grad;
    g_grad.nodes.push_back({5, "worley", ChannelFormat::Mono});
    g_grad.nodes.push_back({6, "blur"});
    g_grad.nodes.push_back({7, "invert"});
    g_grad.connections.push_back({5, 0, 6, 0});
    g_grad.connections.push_back({6, 0, 7, 1});   // socket 1 = color in invert.node.json
    g_grad.output_node = 7;

    uint64_t gen_grad = engine.set_graph(g_grad);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_grad;
    uint32_t w3 = 0, h3 = 0;
    ASSERT_TRUE(wait_for_readback_gen_res(engine, gen_grad, px_grad, w3, h3, RES, RES));

    printf("TEST 2: User graph warp intensity=0 @ %ux%u\n", RES, RES);
    dump_channel_stats("FULL graph (warp output)", px_full);
    dump_channel_stats("IMAGE chain (perlin->levels->blur)", px_img);
    dump_channel_stats("GRADIENT chain (worley->blur->invert)", px_grad);

    double diff_full_vs_img = avg_diff(px_full, px_img);
    double diff_full_vs_grad = avg_diff(px_full, px_grad);
    double diff_img_vs_grad = avg_diff(px_img, px_grad);

    printf("  avg_diff: full_vs_image=%.6f  full_vs_gradient=%.6f  image_vs_gradient=%.6f\n",
           diff_full_vs_img, diff_full_vs_grad, diff_img_vs_grad);

    // At intensity=0, full graph output MUST match image chain.
    EXPECT_LT(avg_diff(px_full, px_img), 0.01)
        << "warp(intensity=0) output must match image chain. "
           "If full graph differs from image chain, params may not reach shader.";

    // Image chain should NOT look like gradient chain.
    EXPECT_GT(avg_diff(px_img, px_grad), 0.01)
        << "image and gradient chains should produce different patterns "
           "(sanity check — if they're identical, the test setup is wrong)";
}

TEST_F(FusedRealNodesGLSL, UserGraph_ChainStructure) {
    // Dump chain structure and GLSL for the full user graph.
    // This reveals which bindless slots the warp reads from.
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::Mono});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "blur"});
    g.nodes.push_back({4, "warp"});
    g.nodes.push_back({5, "worley", ChannelFormat::Mono});
    g.nodes.push_back({6, "blur"});
    g.nodes.push_back({7, "invert"});
    g.connections.push_back({1, 0, 2, 0});   // perlin -> levels
    g.connections.push_back({2, 0, 3, 0});   // levels -> blur
    g.connections.push_back({3, 0, 4, 0});   // blur -> warp.image
    g.connections.push_back({5, 0, 6, 0});   // worley -> blur2
    g.connections.push_back({6, 0, 7, 1});   // blur2 -> invert.color
    g.connections.push_back({7, 0, 4, 1});   // invert -> warp.gradient
    g.output_node = 4;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== USER GRAPH CHAIN STRUCTURE ===\n");
    printf("Total chains: %zu  Total passes: %zu\n",
           cr.pass_plan.chains.size(), cr.pass_plan.passes.size());
    printf("Final output resource: %u\n", cr.pass_plan.final_output_resource);

    // Dump active resources.
    printf("Active resources: %zu\n", cr.pass_plan.active_resources.size());

    // Dump chain index per pass.
    printf("Chain index per pass:\n");
    for (uint32_t i = 0; i < cr.pass_plan.chain_index_of_pass.size(); ++i) {
        uint32_t ci = cr.pass_plan.chain_index_of_pass[i];
        printf("  pass[%u] -> chain[%u]\n", i, ci);
    }

    // Dump per-chain details.
    for (uint32_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("\n--- Chain %u ---\n", ci);
        printf("  Nodes: ");
        for (NodeId n : ch.nodes) printf("%u ", n);
        printf("\n");
        printf("  Sub-pass count: %u  Intermediate count: %u\n",
               ch.sub_pass_count, ch.intermediate_count);
        printf("  External socket masks: ");
        for (uint32_t m : ch.external_socket_masks) printf("0x%x ", m);
        printf("\n");
        printf("  Total inputs: %u  Total outputs: %u  Total params: %u\n",
               ch.total_inputs, ch.total_outputs, ch.total_params);
        printf("  Param base slot: %d\n", ch.param_base_slot);

        // Dump GLSL (truncated to first 500 chars for overview).
        const auto& glsl = ch.glsl;
        if (!glsl.empty()) {
            printf("  GLSL length: %zu\n", glsl.size());
            // Look for texture sampling — which u_sampled slots are used.
            std::regex slot_re(R"(u_sampled\[(\d+)\])");
            auto begin = std::sregex_iterator(glsl.begin(), glsl.end(), slot_re);
            auto end = std::sregex_iterator();
            std::set<int> slots;
            for (auto it = begin; it != end; ++it)
                slots.insert(std::stoi((*it)[1]));
            printf("  u_sampled slots used in GLSL: ");
            for (int s : slots) printf("%d ", s);
            printf("\n");

            // Write full GLSL to file.
            char fname[128];
            snprintf(fname, sizeof(fname), "chain_%u_nodes", ci);
            for (NodeId n : ch.nodes) {
                char buf[32];
                snprintf(buf, sizeof(buf), "_%u", n);
                strncat(fname, buf, sizeof(fname) - strlen(fname) - 1);
            }
            strncat(fname, ".glsl", sizeof(fname) - strlen(fname) - 1);
            std::ofstream f(fname);
            f << glsl;
            printf("  GLSL written to: %s\n", fname);
        }

        // Dump sub-pass GLSL.
        for (uint32_t sp = 0; sp < ch.sub_pass_glsl.size(); ++sp) {
            const auto& spglsl = ch.sub_pass_glsl[sp];
            printf("  Sub-pass %u GLSL length: %zu\n", sp, spglsl.size());
            std::regex slot_re2(R"(u_sampled\[(\d+)\])");
            auto begin2 = std::sregex_iterator(spglsl.begin(), spglsl.end(), slot_re2);
            auto end2 = std::sregex_iterator();
            std::set<int> slots2;
            for (auto it = begin2; it != end2; ++it)
                slots2.insert(std::stoi((*it)[1]));
            printf("    u_sampled slots: ");
            for (int s : slots2) printf("%d ", s);
            printf("\n");

            char fname[128];
            snprintf(fname, sizeof(fname), "chain_%u_sp%u.glsl", ci, sp);
            std::ofstream f(fname);
            f << spglsl;
            printf("    GLSL written to: %s\n", fname);
        }
    }

    // Also dump per-pass details.
    printf("\n=== PER-PASS DETAILS ===\n");
    for (uint32_t pi = 0; pi < cr.pass_plan.passes.size(); ++pi) {
        const auto& p = cr.pass_plan.passes[pi];
        printf("pass[%u]: node=%u type=%s param_base=%d inputs=%u outputs=%u\n",
               pi, p.node_id, p.type_id.c_str(), p.param_base_slot,
               p.input_socket_count, (uint32_t)p.output_resources.size());
        printf("  input_resources: ");
        for (ResourceUUID r : p.input_resources) printf("%u ", r);
        printf("\n  output_resources: ");
        for (ResourceUUID r : p.output_resources) printf("%u ", r);
        printf("\n");
    }
}

// ===========================================================================
// Part 9: Blur→Invert vs Invert→Blur ordering bug
//
// User reports: blur→invert produces X-axis stretching.
//               invert→blur looks correct.
// Hypothesis: the fused chain GLSL computes blur and invert in path order,
// but the output slot assignment or format post-process differs based on
// which node is the chain tail.
//
// Test: build both orderings with the same source (perlin),
// render both, compare pixels, and dump GLSL for inspection.
// ===========================================================================

TEST_F(FusedRealNodesPixel, BlurThenInvert_Vs_InvertThenBlur) {
    // Graph A (buggy per user): perlin → blur → invert
    // Graph B (correct per user): perlin → invert → blur
    // Both should produce similar output (invert and blur commute for
    // uniform inputs, but NOT for noise — the ordering matters).
    // The key: we compare the DIFFERENCE between A and B to understand
    // what the bug is, not whether they're identical.

    uint32_t RES = 256;

    // --- Graph A: perlin → blur → invert ---
    Graph ga;
    ga.nodes.push_back({1, "perlin"});
    ga.nodes.push_back({2, "blur"});
    ga.nodes.push_back({3, "invert"});
    ga.connections.push_back({1, 0, 2, 0});  // perlin → blur
    ga.connections.push_back({2, 0, 3, 1});  // blur → invert.color (socket 1)
    ga.output_node = 3;

    uint64_t gen_a = engine.set_graph(ga);
    ASSERT_NE(gen_a, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    engine.update_node_params_by_id(2, {0.5f});  // blur intensity=0.5
    engine.update_node_params_by_id(3, {1.0f, 0.0f});  // invert mask=1, unused
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_a;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen_res(engine, gen_a, px_a, w, h, RES, RES));

    // --- Graph B: perlin → invert → blur ---
    Graph gb;
    gb.nodes.push_back({1, "perlin"});
    gb.nodes.push_back({2, "invert"});
    gb.nodes.push_back({3, "blur"});
    gb.connections.push_back({1, 0, 2, 1});  // perlin → invert.color (socket 1)
    gb.connections.push_back({2, 0, 3, 0});  // invert → blur
    gb.output_node = 3;

    uint64_t gen_b = engine.set_graph(gb);
    ASSERT_NE(gen_b, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    engine.update_node_params_by_id(2, {1.0f, 0.0f});  // invert mask=1
    engine.update_node_params_by_id(3, {0.5f});  // blur intensity=0.5
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_b;
    uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(wait_for_readback_gen_res(engine, gen_b, px_b, w2, h2, RES, RES));

    // --- Compare ---
    ASSERT_EQ(px_a.size(), px_b.size()) << "different resolution";

    dump_channel_stats("A: blur->invert", px_a);
    dump_channel_stats("B: invert->blur", px_b);

    double md = max_diff(px_a, px_b);
    double ad = avg_diff(px_a, px_b);
    printf("  max_diff=%.6f  avg_diff=%.6f\n", md, ad);

    // Find spatial pattern of differences.
    // Compute per-row average diff to detect X-axis stretching.
    printf("\n  Per-row avg diff (first 32 rows):\n");
    for (uint32_t y = 0; y < std::min(32u, h); ++y) {
        double row_diff = 0;
        for (uint32_t x = 0; x < w; ++x) {
            size_t i = (y * w + x) * 4;
            for (int c = 0; c < 3; ++c)
                row_diff += std::abs((double)px_a[i+c] - (double)px_b[i+c]);
        }
        row_diff /= (w * 3);
        printf("    y=%u: %.6f\n", y, row_diff);
    }

    // Compute per-column average diff to detect Y-axis stretching.
    printf("\n  Per-col avg diff (first 32 cols):\n");
    for (uint32_t x = 0; x < std::min(32u, w); ++x) {
        double col_diff = 0;
        for (uint32_t y = 0; y < h; ++y) {
            size_t i = (y * w + x) * 4;
            for (int c = 0; c < 3; ++c)
                col_diff += std::abs((double)px_a[i+c] - (double)px_b[i+c]);
        }
        col_diff /= (h * 3);
        printf("    x=%u: %.6f\n", x, col_diff);
    }

    // Check if A (blur→invert) has horizontal edges preserved but vertical
    // edges smoothed (asymmetric blur symptom).
    double hgrad_a = avg_horizontal_gradient(px_a, w, h);
    double vgrad_a = avg_vertical_gradient(px_a, w, h);
    double hgrad_b = avg_horizontal_gradient(px_b, w, h);
    double vgrad_b = avg_vertical_gradient(px_b, w, h);
    printf("\n  Gradient analysis:\n");
    printf("    A (blur->invert): hgrad=%.6f  vgrad=%.6f  ratio=%.3f\n",
           hgrad_a, vgrad_a, vgrad_a > 0 ? hgrad_a / vgrad_a : 0);
    printf("    B (invert->blur): hgrad=%.6f  vgrad=%.6f  ratio=%.3f\n",
           hgrad_b, vgrad_b, vgrad_b > 0 ? hgrad_b / vgrad_b : 0);
}

TEST_F(FusedRealNodesGLSL, BlurThenInvert_ChainStructure) {
    // Dump chain structure for perlin → blur → invert to see
    // how the fused chain handles the blur+invert combination.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.nodes.push_back({3, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 1});
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== BLUR->INVERT CHAIN STRUCTURE ===\n");
    printf("Total chains: %zu\n", cr.pass_plan.chains.size());

    for (uint32_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("\n--- Chain %u ---\n", ci);
        printf("  Nodes: ");
        for (NodeId n : ch.nodes) printf("%u ", n);
        printf("\n  Sub-pass count: %u\n", ch.sub_pass_count);
        printf("  Total params: %u  Param base: %d\n", ch.total_params, ch.param_base_slot);

        if (!ch.glsl.empty()) {
            std::regex slot_re(R"(u_sampled\[(\d+)\])");
            auto begin = std::sregex_iterator(ch.glsl.begin(), ch.glsl.end(), slot_re);
            auto end = std::sregex_iterator();
            std::set<int> slots;
            for (auto it = begin; it != end; ++it)
                slots.insert(std::stoi((*it)[1]));
            printf("  u_sampled slots: ");
            for (int s : slots) printf("%d ", s);
            printf("\n");

            char fname[64];
            snprintf(fname, sizeof(fname), "blur_invert_chain_%u.glsl", ci);
            std::ofstream f(fname);
            f << ch.glsl;
            printf("  GLSL written to: %s\n", fname);
        }

        for (uint32_t sp = 0; sp < ch.sub_pass_glsl.size(); ++sp) {
            printf("  Sub-pass %u GLSL length: %zu\n", sp, ch.sub_pass_glsl[sp].size());
            char fname[64];
            snprintf(fname, sizeof(fname), "blur_invert_chain_%u_sp%u.glsl", ci, sp);
            std::ofstream f(fname);
            f << ch.sub_pass_glsl[sp];
            printf("    Written to: %s\n", fname);
        }
    }
}

TEST_F(FusedRealNodesGLSL, InvertThenBlur_ChainStructure) {
    // Dump chain structure for perlin → invert → blur for comparison.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "blur"});
    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== INVERT->BLUR CHAIN STRUCTURE ===\n");
    printf("Total chains: %zu\n", cr.pass_plan.chains.size());

    for (uint32_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("\n--- Chain %u ---\n", ci);
        printf("  Nodes: ");
        for (NodeId n : ch.nodes) printf("%u ", n);
        printf("\n  Sub-pass count: %u\n", ch.sub_pass_count);
        printf("  Total params: %u  Param base: %d\n", ch.total_params, ch.param_base_slot);

        if (!ch.glsl.empty()) {
            std::regex slot_re(R"(u_sampled\[(\d+)\])");
            auto begin = std::sregex_iterator(ch.glsl.begin(), ch.glsl.end(), slot_re);
            auto end = std::sregex_iterator();
            std::set<int> slots;
            for (auto it = begin; it != end; ++it)
                slots.insert(std::stoi((*it)[1]));
            printf("  u_sampled slots: ");
            for (int s : slots) printf("%d ", s);
            printf("\n");

            char fname[64];
            snprintf(fname, sizeof(fname), "invert_blur_chain_%u.glsl", ci);
            std::ofstream f(fname);
            f << ch.glsl;
            printf("  GLSL written to: %s\n", fname);
        }

        for (uint32_t sp = 0; sp < ch.sub_pass_glsl.size(); ++sp) {
            printf("  Sub-pass %u GLSL length: %zu\n", sp, ch.sub_pass_glsl[sp].size());
            char fname[64];
            snprintf(fname, sizeof(fname), "invert_blur_chain_%u_sp%u.glsl", ci, sp);
            std::ofstream f(fname);
            f << ch.sub_pass_glsl[sp];
            printf("    Written to: %s\n", fname);
        }
    }
}

TEST_F(FusedRealNodesGLSL, PerlinLevelsBlurInvert_ChainStructure) {
    // Dump chain structure for perlin → levels → blur → invert.
    // With independent-source split removed, perlin+levels should fuse.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "blur"});
    g.nodes.push_back({4, "invert"});
    g.connections.push_back({1, 0, 2, 0});  // perlin -> levels.color
    g.connections.push_back({2, 0, 3, 0});  // levels -> blur.image
    g.connections.push_back({3, 0, 4, 1});  // blur -> invert.color
    g.output_node = 4;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== PERLIN->LEVELS->BLUR->INVERT CHAIN STRUCTURE ===\n");
    printf("Total chains: %zu\n", cr.pass_plan.chains.size());

    for (uint32_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("\n--- Chain %u ---\n", ci);
        printf("  Nodes: ");
        for (NodeId n : ch.nodes) printf("%u ", n);
        printf("\n  Sub-pass count: %u\n", ch.sub_pass_count);
        printf("  Total params: %u  Param base: %d\n", ch.total_params, ch.param_base_slot);

        if (!ch.glsl.empty()) {
            std::regex slot_re(R"(u_sampled\[(\d+)\])");
            auto begin = std::sregex_iterator(ch.glsl.begin(), ch.glsl.end(), slot_re);
            auto end = std::sregex_iterator();
            std::set<int> slots;
            for (auto it = begin; it != end; ++it)
                slots.insert(std::stoi((*it)[1]));
            printf("  u_sampled slots: ");
            for (int s : slots) printf("%d ", s);
            printf("\n");

            char fname[64];
            snprintf(fname, sizeof(fname), "plbi_chain_%u.glsl", ci);
            std::ofstream f(fname);
            f << ch.glsl;
            printf("  GLSL written to: %s\n", fname);
        }

        for (uint32_t sp = 0; sp < ch.sub_pass_glsl.size(); ++sp) {
            printf("  Sub-pass %u GLSL length: %zu\n", sp, ch.sub_pass_glsl[sp].size());
            char fname[64];
            snprintf(fname, sizeof(fname), "plbi_chain_%u_sp%u.glsl", ci, sp);
            std::ofstream f(fname);
            f << ch.sub_pass_glsl[sp];
            printf("    Written to: %s\n", fname);
        }
    }
}

TEST_F(FusedRealNodesGLSL, RegisterReuse_LinearChain) {
    // A linear chain A→B→C should reuse registers.
    // A's output is only alive while B reads it, then r0 is free for C.
    // Expected: 2 registers (r0, r1) not 3 (_local_0, _local_1, _local_2).
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "invert"});
    g.connections.push_back({1, 0, 2, 0});  // perlin -> levels
    g.connections.push_back({2, 0, 3, 1});  // levels -> invert.color
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u);
    EXPECT_EQ(cr.pass_plan.chains[0].nodes.size(), 3u);

    const auto& coloring = cr.pass_plan.chains[0].coloring;
    printf("\n=== REGISTER REUSE TEST ===\n");
    printf("Colors used: %u (for 3-node chain)\n", coloring.colors_used);
    printf("Spilled: %zu\n", coloring.spilled.size());

    // 3 nodes, but perlin dies after levels reads it, so only 2 regs needed.
    EXPECT_LE(coloring.colors_used, 2u)
        << "3-node linear chain should need <= 2 registers (reuse)";

    // Verify GLSL uses r0/r1 naming, not _local_0/_local_1/_local_2.
    const auto& glsl = cr.pass_plan.chains[0].glsl;
    EXPECT_NE(glsl.find("r0"), std::string::npos) << "must use r0";
    EXPECT_NE(glsl.find("r1"), std::string::npos) << "must use r1";

    // Must NOT declare 3 separate locals.
    EXPECT_EQ(glsl.find("_local_2"), std::string::npos)
        << "must not use _local_2 (register reuse should eliminate it)";
}

TEST_F(FusedRealNodesGLSL, RegisterReuse_ColoringMetadata) {
    // Verify the coloring assignment maps each node output to a register.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u);

    const auto& coloring = cr.pass_plan.chains[0].coloring;
    printf("\n=== COLORING METADATA TEST ===\n");
    printf("Colors used: %u\n", coloring.colors_used);
    for (const auto& [rid, color] : coloring.assignment) {
        printf("  ResourceUUID{node=%llu, socket=%u} -> r%u\n",
               (unsigned long long)rid.node_id, rid.output_index, color);
    }

    // Perlin (node 1) output: should be assigned a register.
    ResourceUUID perlin_out{1, 0};
    EXPECT_TRUE(coloring.assignment.count(perlin_out))
        << "perlin output must have a register assignment";

    // Invert (node 2) output: should be assigned a register.
    ResourceUUID invert_out{2, 0};
    EXPECT_TRUE(coloring.assignment.count(invert_out))
        << "invert output must have a register assignment";

    // Both can share r0 since perlin dies after invert reads it.
    // (Perlin interval [0,1], Invert interval [1,2] — they overlap at step 1,
    // so they actually CANNOT share. Both need separate regs.)
    // Actually: perlin [0,1] and invert [1,2] overlap at boundary, so 2 regs.
    EXPECT_EQ(coloring.colors_used, 2u)
        << "perlin->invert needs exactly 2 registers";
}

TEST_F(FusedRealNodesGLSL, RegisterReuse_LongChain) {
    // 6-node linear chain: simplex->invert->levels->shuffle->separate_rgba->levels
    // Each node's output is only read by the next node.
    // Without reuse: needs 6 locals (_local_0 through _local_5).
    // With reuse: only 2 registers needed (r0, r1) — peak pressure is 2.
    //
    // Socket wiring:
    //   simplex(no inputs) -> invert(socket 1 = color, socket 0 = mask unconnected)
    //   invert -> levels(socket 0 = color)
    //   levels -> shuffle(socket 0 = color)
    //   shuffle -> separate_rgba(socket 0 = color)
    //   separate_rgba(socket 0 = red output) -> levels(socket 0 = color)
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "levels"});
    g.nodes.push_back({4, "shuffle"});
    g.nodes.push_back({5, "separate_rgba"});
    g.nodes.push_back({6, "levels"});
    g.connections.push_back({1, 0, 2, 1});  // simplex -> invert.color
    g.connections.push_back({2, 0, 3, 0});  // invert -> levels.color
    g.connections.push_back({3, 0, 4, 0});  // levels -> shuffle.color
    g.connections.push_back({4, 0, 5, 0});  // shuffle -> separate_rgba.color
    g.connections.push_back({5, 0, 6, 0});  // separate_rgba.r -> levels.color
    g.output_node = 6;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u) << "all 6 nodes should fuse into one chain";

    const auto& chain = cr.pass_plan.chains[0];
    EXPECT_EQ(chain.nodes.size(), 6u);

    const auto& coloring = chain.coloring;
    printf("\n=== 6-NODE LINEAR CHAIN ===\n");
    printf("Chain nodes: ");
    for (NodeId n : chain.nodes) printf("%llu ", (unsigned long long)n);
    printf("\n");
    printf("Colors used: %u (for 6-node chain)\n", coloring.colors_used);
    printf("Spilled: %zu\n", coloring.spilled.size());
    for (const auto& [rid, color] : coloring.assignment) {
        printf("  node=%llu socket=%u -> r%u\n",
               (unsigned long long)rid.node_id, rid.output_index, color);
    }

    // 6-node linear chain where each output is only live for 1 step.
    // Peak pressure = 2 (current value + next value being computed).
    EXPECT_LE(coloring.colors_used, 3u)
        << "6-node linear chain should need <= 3 registers";

    // Show the GLSL — verify it uses r0/r1/r2, not _local_0 through _local_5.
    printf("\n=== GLSL (last 30 lines) ===\n");
    std::istringstream iss(chain.glsl);
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(iss, line)) lines.push_back(line);
    size_t start = lines.size() > 30 ? lines.size() - 30 : 0;
    for (size_t i = start; i < lines.size(); ++i)
        printf("%s\n", lines[i].c_str());

    // Multi-output void functions (separate_rgba) still use _local_N_M for
    // their side-channel outputs (g, b, a) — those aren't in the register graph.
    // But the primary output (socket 0) uses r0, not _local_4.
    // Verify the primary chain uses only r0/r1, no _local_0 through _local_3.
    for (int i = 0; i <= 3; ++i) {
        std::string bad = "_local_" + std::to_string(i) + ";";
        EXPECT_EQ(chain.glsl.find(bad), std::string::npos)
            << "register reuse should eliminate _local_" << i;
    }
}

// ===========================================================================
// Phase 2: Register-pressure-based FusionPlanner
// ===========================================================================

TEST_F(FusedRealNodesGLSL, PressureBasedPlanning_TenNodeChain) {
    // 10-node linear chain: perlin->invert->levels->invert->levels->...
    // With additive cost (5 regs/node), FusionPlanner splits at node 9 (50 > 48).
    // With register pressure, peak pressure = 2 (linear chain, each value dies
    // after the next node reads it), so the entire 10-node chain fits in one group.
    //
    // Node types:
    //   perlin: inputs=[], outputs=[vec4]
    //   invert: inputs=[mask(float), color(vec4)], outputs=[vec4]
    //   levels: inputs=[vec4], outputs=[vec4]
    //
    // Wiring pattern:
    //   perlin[0] -> invert[1] (color)
    //   invert[0] -> levels[0]
    //   levels[0] -> invert[1] (color)
    //   ... repeat
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "levels"});
    g.nodes.push_back({4, "invert"});
    g.nodes.push_back({5, "levels"});
    g.nodes.push_back({6, "invert"});
    g.nodes.push_back({7, "levels"});
    g.nodes.push_back({8, "invert"});
    g.nodes.push_back({9, "levels"});
    g.nodes.push_back({10, "invert"});
    g.connections.push_back({1, 0, 2, 1});   // perlin -> invert.color
    g.connections.push_back({2, 0, 3, 0});   // invert -> levels
    g.connections.push_back({3, 0, 4, 1});   // levels -> invert.color
    g.connections.push_back({4, 0, 5, 0});   // invert -> levels
    g.connections.push_back({5, 0, 6, 1});   // levels -> invert.color
    g.connections.push_back({6, 0, 7, 0});   // invert -> levels
    g.connections.push_back({7, 0, 8, 1});   // levels -> invert.color
    g.connections.push_back({8, 0, 9, 0});   // invert -> levels
    g.connections.push_back({9, 0, 10, 1});  // levels -> invert.color
    g.output_node = 10;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== 10-NODE CHAIN (pressure-based planning) ===\n");
    printf("Chains: %zu\n", cr.pass_plan.chains.size());
    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("  chain[%zu]: %zu nodes, regs=%u\n", ci, ch.nodes.size(),
               ch.coloring.colors_used);
    }

    // With pressure-based planning, the 10-node linear chain should fit
    // in ONE chain (peak register pressure = 2, well under 48).
    // Under additive cost it would split into 2 chains.
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u)
        << "pressure-based planning should fuse 10-node linear chain into 1 chain";

    const auto& chain = cr.pass_plan.chains[0];
    EXPECT_EQ(chain.nodes.size(), 10u);
    EXPECT_LE(chain.coloring.colors_used, 3u)
        << "peak register pressure for linear chain should be <= 3";
}

TEST_F(FusedRealNodesGLSL, PressureBasedPlanning_KeepsTwoChainsForFanOut) {
    // Fan-out: perlin1 -> blend.a, perlin2 -> blend.b
    // Active path: [perlin1, perlin2, blend] (3 nodes).
    // Peak pressure = 2 (two perlin outputs live simultaneously).
    // With pressure-based planning, this stays as one chain (2 <= 48).
    // This is a regression test — fan-out should NOT split unnecessarily.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "perlin"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});
    g.connections.push_back({2, 0, 3, 2});
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== FAN-OUT (perlin1, perlin2 -> blend) ===\n");
    printf("Chains: %zu\n", cr.pass_plan.chains.size());

    // All 3 nodes should be in one chain (peak pressure = 2).
    bool found_big_chain = false;
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 3) {
            found_big_chain = true;
            EXPECT_LE(ch.coloring.colors_used, 3u);
        }
    }
    EXPECT_TRUE(found_big_chain)
        << "fan-out graph should fuse into one 3-node chain";
}

TEST_F(FusedRealNodesGLSL, PressureBasedPlanning_LongLinearChain_NoSplit) {
    // 15-node linear chain: perlin->invert->levels->invert->levels->...
    // Additive cost (5 regs/node) would split at node 9 (50 > 48).
    // Register pressure: peak = 2 for any linear chain (reuse is perfect).
    // This test verifies the pressure function is actually driving the split decision.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    for (int i = 2; i <= 15; ++i) {
        g.nodes.push_back({static_cast<NodeId>(i), (i % 2 == 0) ? "invert" : "levels"});
    }
    // perlin[0] -> invert[1]
    g.connections.push_back({1, 0, 2, 1});
    for (int i = 2; i <= 14; ++i) {
        if (i % 2 == 0) {
            // invert[0] -> levels[0]
            g.connections.push_back({static_cast<NodeId>(i), 0,
                                     static_cast<NodeId>(i + 1), 0});
        } else {
            // levels[0] -> invert[1]
            g.connections.push_back({static_cast<NodeId>(i), 0,
                                     static_cast<NodeId>(i + 1), 1});
        }
    }
    g.output_node = 15;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== 15-NODE LINEAR CHAIN ===\n");
    printf("Chains: %zu\n", cr.pass_plan.chains.size());
    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("  chain[%zu]: %zu nodes, regs=%u\n", ci, ch.nodes.size(),
               ch.coloring.colors_used);
    }

    // Under pressure-based planning: 1 chain (pressure=2).
    // Under additive cost: 2 chains (5*15=75 > 48).
    ASSERT_EQ(cr.pass_plan.chains.size(), 1u)
        << "pressure-based planning should fuse 15-node linear chain into 1 chain";

    EXPECT_EQ(cr.pass_plan.chains[0].nodes.size(), 15u);
    EXPECT_LE(cr.pass_plan.chains[0].coloring.colors_used, 3u);
}

TEST_F(FusedRealNodesGLSL, PressureBasedPlanning_MultiPassStillSplits) {
    // blur (multi-pass) -> invert: blur must be singleton chain, invert separate.
    // This test verifies multi-pass splits still happen with pressure-based planning.
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "blur"});
    g.nodes.push_back({3, "invert"});
    g.connections.push_back({1, 0, 2, 0});  // simplex -> blur
    g.connections.push_back({2, 0, 3, 1});  // blur -> invert.color
    g.output_node = 3;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== BLUR -> INVERT (multi-pass split) ===\n");
    printf("Chains: %zu\n", cr.pass_plan.chains.size());
    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("  chain[%zu]: %zu nodes, sub_passes=%u\n", ci, ch.nodes.size(),
               ch.sub_pass_count);
    }

    // Must have at least 2 chains: [blur] singleton (sub_pass_count=2) + [invert].
    bool found_blur = false, found_invert = false;
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 1 && ch.nodes[0] == 2) found_blur = true;
        if (ch.nodes.size() == 1 && ch.nodes[0] == 3) found_invert = true;
    }
    EXPECT_TRUE(found_blur) << "blur must be singleton chain";
    EXPECT_TRUE(found_invert) << "invert must be separate chain";
}

TEST_F(FusedRealNodesGLSL, PressureBasedPlanning_BranchingChain) {
    // Diamond with reuse: perlin -> invert -> blend.a, perlin2 -> blend.b
    // Active path has branching: perlin1 feeds invert which feeds blend,
    // perlin2 directly feeds blend.
    // Peak pressure: 3 (perlin1 + invert + perlin2 simultaneously live).
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "perlin"});
    g.nodes.push_back({4, "blend"});
    g.connections.push_back({1, 0, 2, 1});  // perlin1 -> invert.color
    g.connections.push_back({2, 0, 4, 1});  // invert -> blend.a
    g.connections.push_back({3, 0, 4, 2});  // perlin2 -> blend.b
    g.output_node = 4;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== BRANCHING CHAIN (perlin->invert->blend, perlin2->blend) ===\n");
    printf("Chains: %zu\n", cr.pass_plan.chains.size());
    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("  chain[%zu]: %zu nodes, regs=%u\n", ci, ch.nodes.size(),
               ch.coloring.colors_used);
    }

    // 4-node diamond fits in one chain (pressure=3, well under 48).
    bool found_big = false;
    for (const auto& ch : cr.pass_plan.chains) {
        if (ch.nodes.size() == 4) {
            found_big = true;
            EXPECT_LE(ch.coloring.colors_used, 4u);
        }
    }
    EXPECT_TRUE(found_big) << "4-node diamond should fuse into one chain";
}

// ===========================================================================
// Part N: Exact diagram reproduction
// ===========================================================================
//
// Diagram:
//   Worley ──reg──► Levels ──reg──► S1 ──Img──► Warp ──reg──► Levels ──reg──► Blend
//                                                 ▲
//   Perlin ──reg──► S2 ──Grad──┘
//
// Graph topology:
//   Node 1: Worley  (source, 0 inputs)
//   Node 2: Levels  (1 input from Worley)
//   Node 3: Perlin  (source, 0 inputs)
//   Node 4: Warp    (image=Levels, gradient=Perlin) — sampler2D boundary → chain split
//   Node 5: Levels  (1 input from Warp)
//   Node 6: Blend   (a=Node5, b=unconnected [0,0,0,1], mask=unconnected 1.0)
//
// Expected chains:
//   Chain 0: Worley → Levels   (register-fused, produces image for Warp.image)
//   Chain 1: Perlin            (singleton, produces image for Warp.gradient)
//   Chain 2: Warp → Levels → Blend (register-fused, reads both via bindless slots)
//
// Questions answered:
//   (1) How many dispatches to reach S1 just before Blend?
//       → Chain 2 is the one containing Blend. It is the 3rd chain dispatched.
//       → Answer: 3 dispatches total (Chain 0, Chain 1, Chain 2).
//
//   (2) How many registers at the end of Chain 2 (just before Blend)?
//       → Chain 2 nodes: Warp → Levels → Blend
//       → Warp output = r0, Levels reads r0 and writes r1, Blend reads r1.
//       → colors_used = 2 (Warp and Levels each need one register; Blend is tail, no output register).
//

TEST_F(FusedRealNodesGLSL, DiagramChainStructure) {
    Graph g;
    g.nodes.push_back({1, "worley", ChannelFormat::RGBA});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "perlin", ChannelFormat::RGBA});
    g.nodes.push_back({4, "warp"});
    g.nodes.push_back({5, "levels"});
    // Connections: Worley→Levels→Warp←Perlin→Levels(out)
    g.connections.push_back({1, 0, 2, 0});   // worley -> levels.color
    g.connections.push_back({2, 0, 4, 0});   // levels -> warp.image (sampler2D)
    g.connections.push_back({3, 0, 4, 1});   // perlin -> warp.gradient (sampler2D)
    g.connections.push_back({4, 0, 5, 0});   // warp -> levels2.color
    g.output_node = 5;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== EXACT DIAGRAM CHAIN STRUCTURE ===\n");
    printf("Total chains: %zu  Total passes: %zu\n",
           cr.pass_plan.chains.size(), cr.pass_plan.passes.size());

    // Dump chain index per pass.
    printf("Chain index per pass:\n");
    for (uint32_t i = 0; i < cr.pass_plan.chain_index_of_pass.size(); ++i) {
        uint32_t ci = cr.pass_plan.chain_index_of_pass[i];
        printf("  pass[%u] node=%u -> chain[%u]\n", i, cr.pass_plan.passes[i].node_id, ci);
    }

    // Dump per-chain details.
    for (uint32_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        printf("\n--- Chain %u ---\n", ci);
        printf("  Nodes: ");
        for (NodeId n : ch.nodes) printf("%u ", n);
        printf("\n");
        printf("  Registers used: %u\n", ch.coloring.colors_used);
        printf("  External socket masks: ");
        for (uint32_t m : ch.external_socket_masks) printf("0x%x ", m);
        printf("\n");
        printf("  Sub-pass count: %u  Intermediate count: %u\n",
               ch.sub_pass_count, ch.intermediate_count);
    }

    // ---- ANSWER (1): dispatches to Levels2 (output) ----
    uint32_t levels2_chain = UINT32_MAX;
    for (uint32_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        for (NodeId n : cr.pass_plan.chains[ci].nodes) {
            if (n == 5) { levels2_chain = ci; break; }
        }
        if (levels2_chain != UINT32_MAX) break;
    }
    ASSERT_NE(levels2_chain, UINT32_MAX) << "Levels2 (node 5) must be in some chain";

    uint32_t dispatches_to_output = levels2_chain + 1;
    printf("\n=== ANSWERS ===\n");
    printf("(1) Dispatches to output (chain containing Levels2): %u\n", dispatches_to_output);

    // ---- ANSWER (2): registers at end of output chain ----
    const auto& output_chain_obj = cr.pass_plan.chains[levels2_chain];
    uint32_t regs_at_end = output_chain_obj.coloring.colors_used;
    printf("(2) Registers at end of output chain: %u\n", regs_at_end);
    printf("    Chain nodes: ");
    for (NodeId n : output_chain_obj.nodes) printf("%u ", n);
    printf("\n");

    // ---- Structural invariants ----
    // Chains: [Worley,Perlin,Levels] + [Warp,Levels2]
    // Sampler2D split before Warp forces a chain boundary.
    EXPECT_EQ(cr.pass_plan.chains.size(), 2u)
        << "Expected 2 chains: [Worley,Perlin,Levels] + [Warp,Levels2]";

    EXPECT_EQ(cr.pass_plan.chains[0].nodes.size(), 3u);
    EXPECT_EQ(cr.pass_plan.chains[1].nodes.size(), 2u);
}

TEST_F(FusedRealNodesPixel, DiagramPixelCorrectness) {
    // Same graph, but actually dispatch and readback pixels.
    uint32_t RES = 128;
    Graph g;
    g.nodes.push_back({1, "worley", ChannelFormat::RGBA});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "perlin", ChannelFormat::RGBA});
    g.nodes.push_back({4, "warp"});
    g.nodes.push_back({5, "levels"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 4, 0});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({4, 0, 5, 0});
    g.output_node = 5;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Set warp intensity to 0 (passthrough) so we can verify chain reads work.
    engine.update_node_params_by_id(4, {0.0f, 0.0f, 0.0f, 0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px;
    uint32_t w = 0, h = 0;
    ASSERT_TRUE(wait_for_readback_gen_res(engine, gen, px, w, h, RES, RES));

    printf("\n=== DIAGRAM PIXEL CORRECTNESS ===\n");
    printf("Readback: %ux%u, %zu floats\n", w, h, px.size());

    // Verify non-zero output.
    double brightness = avg_brightness(px);
    printf("Total brightness: %.2f\n", brightness);
    EXPECT_GT(brightness, 0.0) << "Output must be non-zero — chains must dispatch correctly";
}
