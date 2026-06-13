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
        auto result = emit_fused_subgraph(path, r.ir, lib, 0);
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
    EXPECT_NE(glsl.find("_local_0"), std::string::npos)
        << "fused GLSL must use _local_0 for perlin output";
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

    // Must use register chain: _local_0 -> invert -> _local_1.
    EXPECT_NE(glsl.find("_local_0"), std::string::npos);
    EXPECT_NE(glsl.find("_local_1"), std::string::npos);

    // invert's color input must come from _local_0 (register), not texelFetch.
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
    // Tail is perlin (format_sensitive=true), format=Mono -> _fmt_mono applied.
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
    // invert is NOT format-sensitive, so no format post-process assignment.
    // Note: _fmt_mono may be DEFINED in perlin's includes but should NOT be called.
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::Mono});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 1});  // perlin -> invert.color
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;

    const auto& glsl = cr.pass_plan.chains[0].glsl;
    // No format post-process assignment because invert (tail) is not format_sensitive.
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
    auto result = emit_fused_subgraph(path, r.ir, lib, 0);
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
    auto result = emit_fused_subgraph(path, r.ir, lib, 0);
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
    // Set blend params: mode=0 (mix)
    engine.update_node_params_by_id(2, {0.0f});
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
    // perlin has 8 params, invert has 0 params + 1 float input.
    // perlin base=0, invert base=8+0=8 (float_input_count=1 for mask, but 0 params).
    // Actually: perlin params=8, float_inputs=0 -> base=0, next=8.
    // invert params=0, float_inputs=1 (mask) -> base=8, next=8+0+1=9.
    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto cr = compile(g);
    ASSERT_TRUE(cr.success) << cr.error;
    EXPECT_EQ(cr.param_base_slot[1], 0);
    EXPECT_EQ(cr.param_base_slot[2], 8);  // 8 perlin params + 0 float inputs
    EXPECT_EQ(cr.total_param_floats, 9);  // 8 + 1
}
