// Tests that the blend mode parameter correctly reaches the shader and
// changes the output. Regression guard for param SSBO + chain emission.
#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "test_assets.hpp"
#include <thread>
#include <chrono>
#include <iostream>
#include <fstream>

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
    pc.resolution_x = 128; pc.resolution_y = 128;
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

double mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0; size_t count = 0;
    for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4) {
        sum += std::abs(a[i] - b[i]); ++count;
    }
    return count ? sum / count : -1.0;
}
} // namespace

class BlendModeParam : public ::testing::Test {
protected:
    Engine engine;
    NodeLibrary lib = load_real_lib();
    void SetUp() override {
        if (!init_engine(engine, "test_blend_mode")) GTEST_SKIP() << engine.last_error();
    }
    void TearDown() override {
        engine.shutdown();
    }
};

// Pure-compile dump: shows the GLSL the chain emitter produces, plus the
// SSBO param_base_slot layout. No GPU dispatch -- just inspection.
TEST(BlendModeParamGLSL, DumpChainEmission) {
    NodeLibrary lib = load_real_lib();
    Graph g;
    g.nodes.push_back({1, "color_const"});
    g.nodes.push_back({3, "color_const"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 1});  // cc1 -> blend.a
    g.connections.push_back({3, 0, 2, 2});  // cc2 -> blend.b
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto cr = FusedGraphCompiler::compile(r.ir, lib, 2);
    ASSERT_TRUE(cr.success) << cr.error;

    std::cout << "\n=== param_base_slot map ===" << std::endl;
    for (const auto& kv : cr.param_base_slot) {
        const auto* vn = r.ir.find(kv.first);
        std::cout << "  node " << kv.first << " (" << (vn ? vn->type_id : "?") << ")"
                  << " base=" << kv.second << std::endl;
    }
    std::cout << "  total_param_floats=" << cr.total_param_floats << std::endl;

    std::cout << "\n=== Chains (" << cr.pass_plan.chains.size() << ") ===" << std::endl;
    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        std::cout << "Chain " << ci << " nodes=[";
        for (size_t ni = 0; ni < ch.nodes.size(); ++ni) {
            std::cout << ch.nodes[ni];
            if (ni+1 < ch.nodes.size()) std::cout << ", ";
        }
        std::cout << "] param_base_slot=" << ch.param_base_slot
                  << " total_params=" << ch.total_params << std::endl;
        std::cout << "  param_offsets=[";
        for (size_t i = 0; i < ch.param_offsets.size(); ++i) {
            std::cout << ch.param_offsets[i];
            if (i+1 < ch.param_offsets.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  param_global_slots=[";
        for (size_t i = 0; i < ch.param_global_slots.size(); ++i) {
            std::cout << ch.param_global_slots[i];
            if (i+1 < ch.param_global_slots.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Dump the first chain's GLSL to stdout + file.
    if (!cr.pass_plan.chains.empty()) {
        const auto& glsl = cr.pass_plan.chains[0].glsl;
        std::cout << "\n=== Chain 0 GLSL ===\n" << glsl << std::endl;
        std::ofstream f("blend_chain_dump.glsl");
        f << glsl;
    }
}


// color_const -> blend.a, color_const2 -> blend.b, mask unconnected.
// color_const with mode=0 outputs vec4(r,r,r,1) (grayscale).
// We use non-extreme colors so Mix and HardLight actually diverge.
// With white/black and mask=1.0, Mix gives black (100% b) and HardLight also
// gives black -- mathematically identical. So we use gray tones instead.
TEST_F(BlendModeParam, Mode0_Mix_Vs_Mode14_HardLight_Differ) {
    Graph g;
    g.nodes.push_back({1, "color_const"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 1});  // color_const -> blend.a

    g.nodes.push_back({3, "color_const"});
    g.connections.push_back({3, 0, 2, 2});  // color_const2 -> blend.b

    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // color_const1: gray 0.8 -> blend.a
    engine.update_node_params_by_id(1, {0.0f, 0.8f, 0.8f, 0.8f, 1.0f});
    // color_const2: darker gray 0.3 -> blend.b
    engine.update_node_params_by_id(3, {0.0f, 0.3f, 0.3f, 0.3f, 1.0f});
    // blend: mask=0.5, mode=0 (Mix)
    engine.update_node_params_by_id(2, {0.0f, 0.5f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_mix;
    uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_mix, w, h));
    ASSERT_FALSE(px_mix.empty());

    // Now change ONLY the mode: 0 -> 14 (HardLight). Mode ints must match blend.py.
    engine.update_node_params_by_id(2, {14.0f, 0.5f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px_hardlight;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px_hardlight, w, h));

    double diff = mean_abs_diff(px_mix, px_hardlight);
    std::cout << "[Mode0_Mix vs Mode14_HardLight] mean_abs_diff = " << diff << std::endl;
    std::cout << "  Mix center R: " << px_mix[px_mix.size()/2] << std::endl;
    std::cout << "  HardLight center R: " << px_hardlight[px_hardlight.size()/2] << std::endl;

    // mix(0.8, 0.3, 0.5) = 0.55 for Mix.
    // hardLight(0.8, 0.3) = overlay(0.3, 0.8) = 0.48, then mix(0.8, 0.48, 0.5) = 0.64 for HardLight.
    // These MUST differ.
    EXPECT_GT(diff, 0.01) << "blend mode param has no effect on output";
}

// Same test but verify mask=0 forces passthrough.
// When mask=0, the blend always returns 'a' regardless of mode.
TEST_F(BlendModeParam, Mask0_ForcesPassthrough_RegardlessOfMode) {
    // This test confirms the OPPOSITE: when mask=0, mode changes nothing.
    // It's a control / sanity check on the test methodology.
    Graph g;
    g.nodes.push_back({1, "color_const"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 1});
    g.nodes.push_back({3, "color_const"});
    g.connections.push_back({3, 0, 2, 2});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_TRUE(wait_for_pipeline(engine));

    engine.update_node_params_by_id(1, {0.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    engine.update_node_params_by_id(3, {0.0f, 0.0f, 0.0f, 0.0f, 1.0f});
    // mask=0 explicitly
    engine.update_node_params_by_id(2, {0.0f, 0.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();

    std::vector<float> px1;
    uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px1, w, h));

    engine.update_node_params_by_id(2, {14.0f, 0.0f});  // mode=14 (HardLight), mask still 0
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    engine.poll_pending_compiles();
    std::vector<float> px2;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px2, w, h));

    double diff = mean_abs_diff(px1, px2);
    std::cout << "[mask=0, mode 0 vs 14] mean_abs_diff = " << diff << " (should be ~0)" << std::endl;
    EXPECT_NEAR(diff, 0.0, 0.01) << "mask=0 should force passthrough regardless of mode";
}
