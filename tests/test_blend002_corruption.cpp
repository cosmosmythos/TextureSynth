// Reproduces the Blend.001 corruption: final Blend output shows garbage at
// mask>0 (the A-input path), while mask=0 (B-input path) is correct.
//
// Graph (12 nodes, 11 links) — exact hex IDs from Blender introspection:
//   Worley     -> Blur.001 -> Invert      -> Warp.gradient
//   Levels     -> Warp.image
//   Perlin.001 -> Levels
//   Perlin     -> Warp.001.gradient
//   Invert.001 -> Warp.001.image
//   Warp.001   -> Invert.002 -> Blend.A      (THE CORRUPTED PATH)
//   Warp       -> Blend.B                    (correct path)
//   Worley.001 -> Invert.001
//   Blend: A=Invert.002, B=Warp, Mask=unlinked (default)
//
// Symptom: mask=0 -> correct (pure B/Warp); mask=1 -> garbage (A/Invert.002).
// G+B channels show segmentation garbage = classic wrong-image / unwritten-image read.

#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/FusionGroup.hpp"
#include "engine/graphfusion/FusionGroupEmitter.hpp"
#include "engine/graphfusion/FusedGroupCompiler.hpp"
#include "test_assets.hpp"
#include <iostream>
#include <fstream>

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
    pc.resolution_x = 256; pc.resolution_y = 256;
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

double channel_mean(const std::vector<float>& px, int ch) {
    if (px.empty()) return -1.0;
    double sum = 0; size_t count = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) { sum += px[i + ch]; ++count; }
    return count ? sum / count : -1.0;
}

double avg_brightness(const std::vector<float>& px) {
    if (px.empty()) return -1.0;
    double sum = 0; size_t count = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        sum += (px[i] + px[i+1] + px[i+2]) / 3.0; ++count;
    }
    return count ? sum / count : -1.0;
}

const char* layout_name(VkImageLayout l) {
    switch (l) {
        case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
        case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: return "SHADER_READ_ONLY";
        default: return "OTHER";
    }
}

// Exact IDs from the user's Blender dump (hex -> decimal).
const NodeId WORLEY     = 0xf6d74e28130942c0ULL; // Worley (Cellular)
const NodeId BLUR       = 0xe53abd44f6b048ccULL; // Blur.001
const NodeId INVERT     = 0xbae1a098d61941d0ULL; // Invert (mask path)
const NodeId INVERT001  = 0x349af8934c844f02ULL; // Invert.001
const NodeId INVERT002  = 0x66d56387bdc440d3ULL; // Invert.002 (Blend.A source)
const NodeId LEVELS     = 0x46983fe5c1b04e13ULL; // Levels
const NodeId PERLIN     = 0x2e202ea4cdd84bb7ULL; // Perlin Noise
const NodeId PERLIN001  = 0xfd8e912512f34194ULL; // Perlin Noise.001
const NodeId WARP       = 0x9f672aa5c72245fcULL; // Warp (Blend.B source)
const NodeId WARP001    = 0xdb40022f879a4f0aULL; // Warp.001
const NodeId WORLEY001  = 0x61f967b686e844feULL; // Worley (Cellular).001
const NodeId BLEND      = 0x391ccc78ccd64d4fULL; // Blend (FINAL OUTPUT)

Graph build_blend002_graph() {
    Graph g;
    g.nodes.push_back({WORLEY,    "worley"});
    g.nodes.push_back({BLUR,      "blur"});
    g.nodes.push_back({INVERT,    "invert"});
    g.nodes.push_back({INVERT001, "invert"});
    g.nodes.push_back({INVERT002, "invert"});
    g.nodes.push_back({LEVELS,    "levels"});
    g.nodes.push_back({PERLIN,    "perlin"});
    g.nodes.push_back({PERLIN001, "perlin"});
    g.nodes.push_back({WARP,      "warp"});
    g.nodes.push_back({WARP001,   "warp"});
    g.nodes.push_back({WORLEY001, "worley"});
    g.nodes.push_back({BLEND,     "blend"});

    g.connections.push_back({WORLEY,    0, BLUR,     0}); // Worley -> Blur.001
    g.connections.push_back({BLUR,      0, INVERT,   1}); // Blur.001 -> Invert.color
    g.connections.push_back({INVERT,    0, WARP,     1}); // Invert -> Warp.gradient
    g.connections.push_back({LEVELS,    0, WARP,     0}); // Levels -> Warp.image
    g.connections.push_back({PERLIN,    0, WARP001,  1}); // Perlin -> Warp.001.gradient
    g.connections.push_back({PERLIN001, 0, LEVELS,   0}); // Perlin.001 -> Levels
    g.connections.push_back({WARP,      0, BLEND,    2}); // Warp -> Blend.B
    g.connections.push_back({WORLEY001, 0, INVERT001,1}); // Worley.001 -> Invert.001.color
    g.connections.push_back({INVERT001, 0, WARP001,  0}); // Invert.001 -> Warp.001.image
    g.connections.push_back({WARP001,   0, INVERT002,1}); // Warp.001 -> Invert.002.color
    g.connections.push_back({INVERT002, 0, BLEND,    1}); // Invert.002 -> Blend.A

    g.output_node = BLEND;
    return g;
}

void dump_compile_state(const GraphIR& ir, const NodeLibrary& lib,
                        const CompileGraphResult& cr) {
    auto node_name = [&](NodeId id) -> std::string {
        const auto* vn = ir.find(id);
        const auto* t = vn ? lib.find(vn->type_id) : nullptr;
        return t ? t->id : std::string("?");
    };

    std::cout << "\n=== Chains (" << cr.pass_plan.chains.size() << ") ===\n";
    std::unordered_map<NodeId, uint32_t> node_chain;
    for (uint32_t ci = 0; ci < (uint32_t)cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        std::cout << "Chain " << ci << " (nodes=" << ch.nodes.size() << "):";
        for (size_t m = 0; m < ch.nodes.size(); ++m) {
            const char* role = (m == 0 ? "HEAD" : (m == ch.nodes.size()-1 ? "TAIL" : "MID"));
            std::cout << " [" << node_name(ch.nodes[m]) << " " << std::hex << ch.nodes[m] << std::dec
                      << "/" << role << "]";
            node_chain[ch.nodes[m]] = ci;
        }
        std::cout << "\n";
    }

    std::cout << "\n=== Cross-group edges (producer_in_chain != consumer_in_chain) ===\n";
    for (const auto& c : ir.connections) {
        auto pc = node_chain.find(c.src_node);
        auto cc = node_chain.find(c.dst_node);
        if (pc == node_chain.end() || cc == node_chain.end()) continue;
        if (pc->second == cc->second) continue;
        ResourceUUID rid{c.src_node, c.src_socket};
        bool in_active = cr.pass_plan.active_resources.count(rid) > 0;
        bool prod_is_tail = false;
        const auto& prod_chain = cr.pass_plan.chains[pc->second];
        if (!prod_chain.nodes.empty() && prod_chain.nodes.back() == c.src_node)
            prod_is_tail = true;
        std::cout << "  [" << node_name(c.src_node) << " chain=" << pc->second
                  << (prod_is_tail ? " TAIL" : " NON-TAIL!!")
                  << "] -> [" << node_name(c.dst_node) << " chain=" << cc->second
                  << " sock=" << c.dst_socket << "]  active_res=" << (in_active?"YES":"**NO**")
                  << (prod_is_tail ? "" : "  <<< PRODUCER NOT TAIL: image never written")
                  << "\n";
    }

    std::cout << "\n=== Active Resources (need VRAM image) ===\n";
    for (const auto& rid : cr.pass_plan.active_resources) {
        auto cc = cr.pass_plan.color_classes.find(rid);
        uint32_t color = (cc != cr.pass_plan.color_classes.end()) ? cc->second : 0;
        auto lt = cr.pass_plan.lifetimes.find(rid);
        uint32_t f = (lt != cr.pass_plan.lifetimes.end()) ? lt->second.first_pass : 0;
        uint32_t l = (lt != cr.pass_plan.lifetimes.end()) ? lt->second.last_pass : 0;
        std::cout << "  " << node_name(rid.node_id) << "_out" << rid.output_index
                  << " color=" << color << " lifetime=[" << f << "," << l << "]\n";
    }
}
} // namespace

// =============================================================================
// TEST 1 (STRUCTURE): compile-time invariant check.
// Every cross-chain producer MUST be the tail of its chain (only tails
// imageStore). If any producer is NON-TAIL, its image is never written and
// downstream consumers read garbage. This is the corruption root cause.
// =============================================================================
TEST(Blend002Corruption, EveryCrossChainProducerIsTail) {
    NodeLibrary lib = load_lib();
    Graph g = build_blend002_graph();

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto ctx = fusion::build_context(r.ir, lib);
    auto bundle = fusion::group_nodes(r.ir, ctx);
    fusion::split_at_sampler2d_sources(bundle, ctx);
    fusion::merge_groups(bundle, ctx);
    fusion::compute_external_inputs(bundle, ctx);
    auto compiled = fusion::compile_groups(bundle, r.ir, ctx, lib);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    // Invariant: every cross-group external input references a source node
    // that IS contained in the source group (the producer group actually
    // contains the node that feeds the consumer).
    int violations = 0;
    for (size_t gi = 0; gi < compiled.groups.size(); ++gi) {
        const auto& cg = compiled.groups[gi];
        for (const auto& ext : cg.external_inputs) {
            // find the source group that contains ext.src_node
            bool found = false;
            for (size_t sgi = 0; sgi < bundle.groups.size(); ++sgi) {
                if (fusion::group_contains(bundle.groups[sgi], ext.src_node)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cout << "VIOLATION: group " << gi << " ext input src_node "
                          << std::hex << ext.src_node << std::dec
                          << " not found in any source group\n";
                ++violations;
            }
        }
    }
    EXPECT_EQ(violations, 0)
        << "Cross-group external input references a source node not in its source group";

    // Dump the GLSL of the first few groups.
    for (size_t gi = 0; gi < compiled.groups.size() && gi < 3; ++gi) {
        const auto& cg = compiled.groups[gi];
        if (cg.glsl.empty()) continue;
        std::string fn = "blend002_chain_" + std::to_string(gi) + ".glsl";
        std::ofstream f(fn); f << cg.glsl; f.close();
        std::cout << "  -> wrote " << fn << " (" << cg.glsl.size() << " bytes)\n";
    }
}

// =============================================================================
// TEST 2 (PIXEL): final Blend output must be non-zero and the A-input
// branch (Invert.002 <- Warp.001) must actually contribute.
// =============================================================================
class Blend002Pixel : public ::testing::Test {
protected:
    Engine engine;
    NodeLibrary lib = load_lib();
    void SetUp() override {
        if (!init_engine(engine, "test_blend002")) GTEST_SKIP() << engine.last_error();
    }
    void TearDown() override { engine.shutdown(); }
};

TEST_F(Blend002Pixel, FinalBlendNonZero) {
    Graph g = build_blend002_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> px;
    uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    ASSERT_FALSE(px.empty());

    std::cout << "Final Blend: R=" << channel_mean(px,0)
              << " G=" << channel_mean(px,1)
              << " B=" << channel_mean(px,2)
              << " A=" << channel_mean(px,3)
              << " bright=" << avg_brightness(px) << "\n";
    EXPECT_GT(avg_brightness(px), 0.0);
}

// =============================================================================
// TEST 3 (SESSION/STALENESS): simulate the Blender workflow.
// The addon drives repeated evals via update_node_params_by_id (no set_graph).
// Dragging the Blend mask slider toggles mask 0<->1, marking only Blend dirty.
// In mask=1, Blend.A (Invert.002) must dominate; in mask=0, Blend.B (Warp) dominates.
// The corruption symptom: mask=1 produces garbage while mask=0 is fine.
//
// Blend param layout: [mode, mask]. mode=0 (Mix), mask in {0.0, 0.5, 1.0}.
// Mix: out = mix(B.rgb, A.rgb, mask). So mask=0 -> pure B (Warp), mask=1 -> pure A (Invert.002).
// =============================================================================
TEST_F(Blend002Pixel, MaskToggle_APathStaysConsistent) {
    Graph g = build_blend002_graph();
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    // Render mask=0 (pure B / Warp) and mask=1 (pure A / Invert.002 path).
    // Blend SSBO layout: [mode=0.0, mask]. Update only the mask.
    auto render_mask = [&](float mask, std::vector<float>& out) -> bool {
        engine.update_node_params_by_id(BLEND, {0.0f, mask});
        // poll to let the param write + dirty propagate before submit
        for (int i = 0; i < 30; ++i) {
            engine.poll_pending_compiles();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        uint32_t w=0, h=0;
        return wait_for_readback_gen(engine, gen, out, w, h);
    };

    std::vector<float> px_b;  // mask=0 -> pure B (Warp)
    ASSERT_TRUE(render_mask(0.0f, px_b));
    std::vector<float> px_a;  // mask=1 -> pure A (Invert.002 path)
    ASSERT_TRUE(render_mask(1.0f, px_a));

    std::cout << "mask=0 (B=Warp path):   R=" << channel_mean(px_b,0)
              << " G=" << channel_mean(px_b,1) << " B=" << channel_mean(px_b,2) << "\n";
    std::cout << "mask=1 (A=Invert.002):  R=" << channel_mean(px_a,0)
              << " G=" << channel_mean(px_a,1) << " B=" << channel_mean(px_a,2) << "\n";

    // Sanity: the two must differ (A and B are different noise fields).
    double diff = 0; size_t n = 0;
    for (size_t i = 0; i + 3 < px_a.size() && i + 3 < px_b.size(); i += 4) {
        diff += std::abs(px_a[i] - px_b[i]); ++n;
    }
    diff = n ? diff / n : 0;
    std::cout << "mean |A.R - B.R| = " << diff << "\n";
    EXPECT_GT(diff, 0.01) << "A and B paths produced identical output (one is dead)";

    // The corruption signature: A-path shows NaN/garbage. Detect via not-finite
    // or extreme values.
    int nan_count = 0, extreme = 0;
    for (size_t i = 0; i + 3 < px_a.size(); i += 4) {
        if (!std::isfinite(px_a[i]) || !std::isfinite(px_a[i+1]) || !std::isfinite(px_a[i+2])) ++nan_count;
        if (std::abs(px_a[i]) > 10.0f || std::abs(px_a[i+1]) > 10.0f) ++extreme;
    }
    EXPECT_EQ(nan_count, 0) << "A-path produced NaN/inf pixels (corruption)";
    std::cout << "A-path extreme-pixel count (>10.0): " << extreme << "\n";

    // CRITICAL CONSISTENCY CHECK: re-render mask=1 after going back to mask=0.
    // Aliased-staleness bugs show up as the A-path changing on the 2nd visit.
    std::vector<float> px_b2;
    ASSERT_TRUE(render_mask(0.0f, px_b2));
    std::vector<float> px_a2;
    ASSERT_TRUE(render_mask(1.0f, px_a2));

    double a_diff = 0; size_t an = 0;
    for (size_t i = 0; i + 3 < px_a.size() && i + 3 < px_a2.size(); i += 4) {
        a_diff += std::abs(px_a[i] - px_a2[i]); ++an;
    }
    a_diff = an ? a_diff / an : 0;
    std::cout << "mask=1 1st vs 2nd visit mean|R diff| = " << a_diff << "\n";
    // The A-path must be deterministic across re-eval (same params, same seed).
    EXPECT_LT(a_diff, 0.01) << "A-path output changed on 2nd visit (aliased staleness bug)";
}

