#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/PassPlan.hpp"
#include "test_assets.hpp"
#include <fstream>
#include <thread>
#include <chrono>
#include <cmath>
#include <string>

using namespace te;

namespace {

NodeLibrary load_lib() {
    NodeLibrary lib;
    std::string err;
    int n = NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    EXPECT_GT(n, 0) << "failed to load nodes: " << err;
    return lib;
}

bool init_engine(Engine& e, const char* cache_name) {
    return e.init(VK_NULL_HANDLE, nullptr, 0, true, cache_name,
                  find_test_nodes_dir().c_str(),
                  find_test_glsl_dir().c_str());
}

bool wait_for_pipeline(Engine& e, int timeout_ms = 3000) {
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        e.poll_pending_compiles();
        if (e.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return e.has_pipeline();
}

bool wait_readback(Engine& e, uint64_t gen, std::vector<float>& px,
                   uint32_t& w, uint32_t& h, int timeout_ms = 5000) {
    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = e.async_readback().submit(e.ctx(), e, pc, gen);
    if (ticket == 0) return false;
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        uint64_t og = 0;
        if (e.async_readback().poll(e.ctx(), px, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

} // anonymous namespace

// ============================================================================
// TEST 1: Inspect Chain struct — verify sub-pass GLSL content
// Build image→blur graph, check the Chain has 2 sub-passes, dump GLSL to files.
// ============================================================================
TEST(BlurDebug, InspectChainSubPassGLSL) {
    NodeLibrary lib = load_lib();
    Engine engine;
    ASSERT_TRUE(init_engine(engine, "test_blur_debug_inspect"));
    printf("[test-debug] init OK\n");

    Graph g;
    g.nodes.push_back({1, "image"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    printf("[test-debug] set_graph gen=%llu err=%s\n", (unsigned long long)gen, engine.last_error());
    ASSERT_NE(gen, 0u) << engine.last_error();
    engine.set_resolution(64, 64);
    printf("[test-debug] set_resolution OK\n");
    ASSERT_TRUE(wait_for_pipeline(engine));
    printf("[test-debug] wait_for_pipeline OK\n");

    printf("[test-debug] inspecting chains...\n");
    // For deeper inspection, we build the graph through GraphCompiler manually.
    auto ir_result = validate_graph(g, lib);
    ASSERT_TRUE(ir_result.success) << ir_result.error;

    resolve_node_depths(ir_result.ir);
    auto compile_result = GraphCompiler::compile(ir_result.ir, lib);
    ASSERT_TRUE(compile_result.success) << compile_result.error;
    const auto& plan = compile_result.pass_plan;

    printf("\n=== Chain Inspection ===\n");
    printf("Number of chains: %zu\n", plan.chains.size());

    for (size_t ci = 0; ci < plan.chains.size(); ++ci) {
        const auto& chain = plan.chains[ci];
        printf("Chain %zu: sub_pass_count=%u intermediate_count=%u nodes=",
               ci, chain.sub_pass_count, chain.intermediate_count);
        for (NodeId n : chain.nodes) printf(" %llu", (unsigned long long)n);
        printf("\n");

        if (chain.sub_pass_count > 1) {
            printf("  sub_pass_glsl size: %zu\n", chain.sub_pass_glsl.size());
            for (size_t sp = 0; sp < chain.sub_pass_glsl.size(); ++sp) {
                const std::string& glsl = chain.sub_pass_glsl[sp];
                printf("  Sub-pass %zu: GLSL length=%zu chars\n", sp, glsl.size());

                // Dump GLSL to file for inspection.
                std::string filename = "blur_debug_sp" + std::to_string(sp) + ".glsl";
                std::ofstream f(filename);
                f << glsl;
                printf("  Dumped to: %s\n", filename.c_str());

                // Check for ts_pass_index specialization constant.
                bool has_spec = glsl.find("constant_id = 0") != std::string::npos;
                printf("  Has specialization constant (constant_id=0): %s\n",
                       has_spec ? "YES" : "NO");

                // Check for ts_pass_index usage.
                bool uses_ts_pass = glsl.find("ts_pass_index") != std::string::npos;
                printf("  Uses ts_pass_index: %s\n", uses_ts_pass ? "YES" : "NO");

                // Check which direction branch is present.
                bool has_H = glsl.find("vec2(sigma, 0.0)") != std::string::npos;
                bool has_V = glsl.find("vec2(0.0, sigma)") != std::string::npos;
                printf("  Has H direction (sigma,0): %s\n", has_H ? "YES" : "NO");
                printf("  Has V direction (0,sigma): %s\n", has_V ? "YES" : "NO");
            }

            // Check variant keys.
            printf("  sub_pass_variant_keys size: %zu\n", chain.sub_pass_variant_keys.size());
            for (size_t sp = 0; sp < chain.sub_pass_variant_keys.size(); ++sp) {
                const auto& key = chain.sub_pass_variant_keys[sp];
                printf("  Key sp=%zu: node_type_id=%s spec[0]=%u spec_count=%u\n",
                       sp, key.node_type_id.c_str(),
                       key.specialization[0], key.specialization_count);
            }
        }
    }

    printf("[test-debug] about to shutdown\n");
    engine.shutdown();
    printf("[test-debug] shutdown done\n");
}

// ============================================================================
// TEST 2: Low-intensity gradient measurement — the actual bug exposer
// Creates a smooth gradient, applies blur at intensity=0.03, measures H vs V
// gradient reduction independently.
// ============================================================================
TEST(BlurDebug, LowIntensityGradientMeasurement) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine, "test_blur_debug_gradient"));

    constexpr uint32_t W = 64, H = 64;

    // --- Test A: Horizontal gradient (measures V-blur effect) ---
    // Pixel value = x/W, so gradient goes left-to-right.
    // V-blur does NOT affect horizontal gradients (edges are horizontal lines).
    // H-blur DOES affect horizontal gradients (edges are vertical lines).
    {
        std::vector<float> px(W * H * 4);
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                float val = (float)x / (float)W;
                uint32_t i = (y * W + x) * 4;
                px[i + 0] = val; px[i + 1] = val; px[i + 2] = val; px[i + 3] = 1.0f;
            }
        }

        const uint64_t image_id = 20;
        ASSERT_TRUE(engine.upload_image(image_id, px.data(), W, H));
        for (int i = 0; i < 200; ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        Graph g;
        g.nodes.push_back({image_id, "image"});
        g.nodes.push_back({2, "blur"});
        g.connections.push_back({image_id, 0, 2, 0});
        g.output_node = 2;

        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        engine.set_resolution(W, H);
        ASSERT_TRUE(wait_for_pipeline(engine));

        engine.update_node_params_by_id(2, {0.03f});
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

        // Measure horizontal gradient (sum of |out[x+1] - out[x]| averaged over all rows).
        double hgrad_blurred = 0.0;
        double hgrad_original = 0.0;
        for (uint32_t y = 0; y < oh; ++y) {
            for (uint32_t x = 0; x + 1 < ow; ++x) {
                size_t i0 = (y * ow + x) * 4;
                size_t i1 = (y * ow + x + 1) * 4;
                hgrad_blurred += std::abs(out[i1] - out[i0]);
                hgrad_original += 1.0 / W;  // original gradient is uniform
            }
        }
        hgrad_blurred /= (oh * (ow - 1));
        hgrad_original /= (oh * (ow - 1));

        printf("\n=== HORIZONTAL GRADIENT (measures H-blur effect) ===\n");
        printf("Original gradient: %.6f\n", hgrad_original);
        printf("Blurred gradient:  %.6f\n", hgrad_blurred);
        printf("Reduction: %.1f%%\n", 100.0 * (1.0 - hgrad_blurred / hgrad_original));

        // Print center row for inspection.
        printf("Row y=%u after blur:\n", oh / 2);
        for (uint32_t x = 0; x < ow; x += 8) {
            size_t i = (oh / 2 * ow + x) * 4;
            printf("  x=%2u: R=%.6f (expect ~%.3f)\n", x, out[i], (float)x / W);
        }
    }

    engine.shutdown();
    ASSERT_TRUE(init_engine(engine, "test_blur_debug_gradient2"));

    // --- Test B: Vertical gradient (measures V-blur effect) ---
    {
        std::vector<float> px(W * H * 4);
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                float val = (float)y / (float)H;
                uint32_t i = (y * W + x) * 4;
                px[i + 0] = val; px[i + 1] = val; px[i + 2] = val; px[i + 3] = 1.0f;
            }
        }

        const uint64_t image_id = 21;
        ASSERT_TRUE(engine.upload_image(image_id, px.data(), W, H));
        for (int i = 0; i < 200; ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        Graph g;
        g.nodes.push_back({image_id, "image"});
        g.nodes.push_back({2, "blur"});
        g.connections.push_back({image_id, 0, 2, 0});
        g.output_node = 2;

        uint64_t gen = engine.set_graph(g);
        ASSERT_NE(gen, 0u) << engine.last_error();
        engine.set_resolution(W, H);
        ASSERT_TRUE(wait_for_pipeline(engine));

        engine.update_node_params_by_id(2, {0.03f});
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

        // Measure vertical gradient (sum of |out[y+1] - out[y]| averaged over all columns).
        double vgrad_blurred = 0.0;
        double vgrad_original = 0.0;
        for (uint32_t y = 0; y + 1 < oh; ++y) {
            for (uint32_t x = 0; x < ow; ++x) {
                size_t i0 = (y * ow + x) * 4;
                size_t i1 = ((y + 1) * ow + x) * 4;
                vgrad_blurred += std::abs(out[i1] - out[i0]);
                vgrad_original += 1.0 / H;
            }
        }
        vgrad_blurred /= ((oh - 1) * ow);
        vgrad_original /= ((oh - 1) * ow);

        printf("\n=== VERTICAL GRADIENT (measures V-blur effect) ===\n");
        printf("Original gradient: %.6f\n", vgrad_original);
        printf("Blurred gradient:  %.6f\n", vgrad_blurred);
        printf("Reduction: %.1f%%\n", 100.0 * (1.0 - vgrad_blurred / vgrad_original));

        // Print center column for inspection.
        printf("Column x=%u after blur:\n", ow / 2);
        for (uint32_t y = 0; y < oh; y += 8) {
            size_t i = (y * ow + ow / 2) * 4;
            printf("  y=%2u: R=%.6f (expect ~%.3f)\n", y, out[i], (float)y / H);
        }
    }

    engine.shutdown();
}

// ============================================================================
// TEST 3: Full pipeline — image → Levels → blur at low intensity
// Reproduces the user's exact Blender graph topology.
// ============================================================================
TEST(BlurDebug, LevelsBlurPipelineLowIntensity) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine, "test_blur_debug_levels_blur"));

    constexpr uint32_t W = 64, H = 64;

    std::vector<float> px(W * H * 4);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            float val = (float)x / (float)W;
            uint32_t i = (y * W + x) * 4;
            px[i + 0] = val; px[i + 1] = val; px[i + 2] = val; px[i + 3] = 1.0f;
        }
    }

    const uint64_t image_id = 30;
    ASSERT_TRUE(engine.upload_image(image_id, px.data(), W, H));
    for (int i = 0; i < 200; ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // image → levels → blur (matching user's Blender tree)
    Graph g;
    g.nodes.push_back({image_id, "image"});
    g.nodes.push_back({3, "levels"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({image_id, 0, 3, 0});
    g.connections.push_back({3, 0, 2, 0});
    g.output_node = 2;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    engine.set_resolution(W, H);
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Set blur intensity=0.03
    engine.update_node_params_by_id(2, {0.03f});
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

    // Compute both gradient reductions.
    double hgrad_blurred = 0.0, hgrad_orig = 0.0;
    for (uint32_t y = 0; y < oh; ++y)
        for (uint32_t x = 0; x + 1 < ow; ++x) {
            size_t i0 = (y * ow + x) * 4;
            size_t i1 = (y * ow + x + 1) * 4;
            hgrad_blurred += std::abs(out[i1] - out[i0]);
            hgrad_orig += 1.0 / W;
        }
    hgrad_blurred /= (oh * (ow - 1));
    hgrad_orig /= (oh * (ow - 1));

    double vgrad_blurred = 0.0, vgrad_orig = 0.0;
    for (uint32_t y = 0; y + 1 < oh; ++y)
        for (uint32_t x = 0; x < ow; ++x) {
            size_t i0 = (y * ow + x) * 4;
            size_t i1 = ((y + 1) * ow + x) * 4;
            vgrad_blurred += std::abs(out[i1] - out[i0]);
            vgrad_orig += 1.0 / H;
        }
    vgrad_blurred /= ((oh - 1) * ow);
    vgrad_orig /= ((oh - 1) * ow);

    double hreduction = 100.0 * (1.0 - hgrad_blurred / hgrad_orig);
    double vreduction = 100.0 * (1.0 - vgrad_blurred / vgrad_orig);

    printf("\n=== LEVELS + BLUR (intensity=0.03) ===\n");
    printf("H-blur gradient reduction: %.1f%%\n", hreduction);
    printf("V-blur gradient reduction: %.1f%%\n", vreduction);
    printf("Ratio V/H: %.2f\n", vreduction / std::max(hreduction, 0.001));

    printf("Row y=%u:\n", oh / 2);
    for (uint32_t x = 0; x < ow; x += 8) {
        size_t i = (oh / 2 * ow + x) * 4;
        printf("  x=%2u: R=%.6f\n", x, out[i]);
    }
    printf("Column x=%u:\n", ow / 2);
    for (uint32_t y = 0; y < oh; y += 8) {
        size_t i = (y * ow + ow / 2) * 4;
        printf("  y=%2u: R=%.6f\n", y, out[i]);
    }

    engine.shutdown();
}

// ============================================================================
// TEST 4: Debug logging for blur→blend fusion merge.
// Graph: image(1) → blur(2) → blend(3). Tests:
//   - merge_groups: is blur(pass1) merged with blend?
//   - populate_groups_: can pass 1 find the intermediate?
//   - Fallback scan: which group's output does blend read for blur's input?
// ============================================================================
TEST(BlurDebug, FusedBlurThenBlendMergeTracing) {
    printf("\n===== TEST: image->blur->blend, output=blend =====\n");
    NodeLibrary lib = load_lib();
    Engine engine;
    ASSERT_TRUE(init_engine(engine, "test_blur_debug_blur_blend"));

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "blur"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 2, 0});   // simplex out -> blur.tex (Sampler2D)
    g.connections.push_back({2, 0, 3, 1});   // blur out -> blend.a (Vec4)
    g.connections.push_back({2, 0, 3, 2});   // blur out -> blend.b (Vec4)
    g.output_node = 3;                        // blend is active

    printf("[test] set_resolution(128,128)\n");
    engine.set_resolution(128, 128);
    printf("[test] set_graph()...\n");
    uint64_t gen = engine.set_graph(g);
    printf("[test] set_graph gen=%llu err=%s\n", (unsigned long long)gen, engine.last_error());
    ASSERT_NE(gen, 0u) << engine.last_error();

    printf("[test] waiting for pipeline...\n");
    EXPECT_TRUE(wait_for_pipeline(engine));
    printf("[test] pipeline ready\n");

    engine.shutdown();

    // --- Second run: blur as output for comparison ---
    printf("\n===== TEST: image->blur->blend, output=blur =====\n");
    Engine engine2;
    ASSERT_TRUE(init_engine(engine2, "test_blur_debug_blur_blend_out2"));
    Graph g2;
    g2.nodes.push_back({1, "simplex"});
    g2.nodes.push_back({2, "blur"});
    g2.nodes.push_back({3, "blend"});
    g2.connections.push_back({1, 0, 2, 0});   // simplex out -> blur.tex (Sampler2D)
    g2.connections.push_back({2, 0, 3, 1});   // blur out -> blend.a (Vec4)
    g2.connections.push_back({2, 0, 3, 2});   // blur out -> blend.b (Vec4)
    g2.output_node = 2;  // blur is active

    printf("[test] set_resolution(128,128)\n");
    engine2.set_resolution(128, 128);
    printf("[test] set_graph()...\n");
    uint64_t gen2 = engine2.set_graph(g2);
    printf("[test] set_graph gen=%llu err=%s\n", (unsigned long long)gen2, engine2.last_error());
    ASSERT_NE(gen2, 0u) << engine2.last_error();

    printf("[test] waiting for pipeline...\n");
    EXPECT_TRUE(wait_for_pipeline(engine2));
    printf("[test] pipeline ready\n");

    engine2.shutdown();
}
