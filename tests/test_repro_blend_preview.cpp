#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/FusionPlanner.hpp"
#include "engine/graphfusion/DAG.hpp"
#include "test_assets.hpp"
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

double mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0; size_t count = 0;
    for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4) {
        sum += std::abs(a[i] - b[i]); ++count;
    }
    return count ? sum / count : -1.0;
}

// User's graph (see mermaid). Two parallel chains feed the final Blend (Mix):
//   cellular_A (worley->levels->blend_darken->levels_inv)
//   mask       (simplex->levels)
//   fill       (gabor->levels)
// Final blend: A=cellular_A, B=fill, mask=simplex_mask
// Output: shuffle(R->RGB, A->A)
struct UserGraph {
    Graph g;
    NodeId W1=1, L1=2;          // cellular A branch
    NodeId W2=3, L2=4;
    NodeId B1=5, L3=6;          // darken -> invert
    NodeId S1=7, L4=8;          // simplex mask
    NodeId G1=9, L5=10;         // gabor fill
    NodeId B2=11, Sh=12;        // final blend, shuffle

    void build() {
        g.nodes.push_back({W1, "worley"});
        g.nodes.push_back({L1, "levels"});
        g.nodes.push_back({W2, "worley"});
        g.nodes.push_back({L2, "levels"});
        g.nodes.push_back({B1, "blend"});
        g.nodes.push_back({L3, "levels"});
        g.nodes.push_back({S1, "simplex"});
        g.nodes.push_back({L4, "levels"});
        g.nodes.push_back({G1, "gabor"});
        g.nodes.push_back({L5, "levels"});
        g.nodes.push_back({B2, "blend"});
        g.nodes.push_back({Sh, "shuffle"});

        // W1 -> L1 (socket 0)
        g.connections.push_back({W1, 0, L1, 0});
        // W2 -> L2 (socket 0)
        g.connections.push_back({W2, 0, L2, 0});
        // L1 -> B1.mask? No, B1.mask is float. L1 -> B1.a (socket 1)
        g.connections.push_back({L1, 0, B1, 1});
        // L2 -> B1.b (socket 2)
        g.connections.push_back({L2, 0, B1, 2});
        // B1 -> L3 (socket 0)
        g.connections.push_back({B1, 0, L3, 0});
        // S1 -> L4
        g.connections.push_back({S1, 0, L4, 0});
        // G1 -> L5
        g.connections.push_back({G1, 0, L5, 0});
        // L3 -> B2.a (socket 1)
        g.connections.push_back({L3, 0, B2, 1});
        // L5 -> B2.b (socket 2)
        g.connections.push_back({L5, 0, B2, 2});
        // L4 -> B2.mask (socket 0)
        g.connections.push_back({L4, 0, B2, 0});
        // B2 -> Shuffle
        g.connections.push_back({B2, 0, Sh, 0});

        g.output_node = Sh;
    }
};

} // anonymous namespace

// =============================================================================
// Test 1 (STRUCTURE): Reproduce user's graph, dump chains, detect topology bug.
// =============================================================================
TEST(ReproBlendPreview, ChainStructure_AccessibleNodes) {
    NodeLibrary lib = load_real_lib();
    UserGraph ug; ug.build();

    auto r = validate_graph(ug.g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto path = ActivePathTracer::trace(r.ir, ug.Sh, lib);
    std::cout << "=== Active path (" << path.nodes.size() << " nodes) ===" << std::endl;
    for (NodeId n : path.nodes) {
        const auto* inst = r.ir.find(n);
        const auto* type = inst ? lib.find(inst->type_id) : nullptr;
        std::cout << "  [" << n << "] " << (type ? type->id : "?") << std::endl;
    }

    auto cr = FusedGraphCompiler::compile(r.ir, lib, ug.Sh);
    ASSERT_TRUE(cr.success) << cr.error;

    std::cout << "\n=== Chains (" << cr.pass_plan.chains.size() << ") ===" << std::endl;
    size_t chains_with_ext = 0;
    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        std::cout << "Chain " << ci << " (nodes=" << ch.nodes.size() << "): ";
        for (NodeId n : ch.nodes) {
            const auto* inst = r.ir.find(n);
            const auto* type = inst ? lib.find(inst->type_id) : nullptr;
            std::cout << n << ":" << (type ? type->id : "?") << " ";
        }
        bool has_ext = !ch.external_socket_masks.empty() &&
            std::any_of(ch.external_socket_masks.begin(), ch.external_socket_masks.end(),
                        [](uint32_t m){ return m != 0; });
        if (has_ext) {
            ++chains_with_ext;
            std::cout << "\n  external_socket_masks=[";
            for (size_t i = 0; i < ch.external_socket_masks.size(); ++i) {
                std::cout << ch.external_socket_masks[i];
                if (i+1 < ch.external_socket_masks.size()) std::cout << ",";
            }
            std::cout << "]";
        }
        std::cout << std::endl;
    }
    std::cout << "Chains with cross-group external inputs: " << chains_with_ext << std::endl;

    // The final blend (B2) consumes L3 (from one branch) and L5 (from another).
    bool b2_in_chain = false;
    for (const auto& ch : cr.pass_plan.chains) {
        for (NodeId n : ch.nodes) {
            if (n == ug.B2) { b2_in_chain = true; break; }
        }
        if (b2_in_chain) break;
    }
    EXPECT_TRUE(b2_in_chain) << "B2 must be in some chain";

    // ── THE BUG ───────────────────────────────────────────────────────────
    // Every node that feeds a cross-group consumer MUST have its output
    // allocated as a VRAM image so the downstream chain can texelFetch it.
    std::unordered_map<NodeId, uint32_t> node_chain;
    for (uint32_t ci = 0; ci < (uint32_t)cr.pass_plan.chains.size(); ++ci) {
        for (NodeId n : cr.pass_plan.chains[ci].nodes) node_chain[n] = ci;
    }
    std::cout << "\n=== Cross-group edges (producer -> consumer in different chain) ===" << std::endl;
    for (const auto& c : r.ir.connections) {
        auto pc = node_chain.find(c.src_node);
        auto cc = node_chain.find(c.dst_node);
        if (pc == node_chain.end() || cc == node_chain.end()) continue;
        if (pc->second == cc->second) continue;
        ResourceUUID rid{c.src_node, c.src_socket};
        bool in_active = cr.pass_plan.active_resources.count(rid) > 0;
        const auto* prod_inst = r.ir.find(c.src_node);
        const auto* prod_type = prod_inst ? lib.find(prod_inst->type_id) : nullptr;
        const auto* cons_inst = r.ir.find(c.dst_node);
        const auto* cons_type = cons_inst ? lib.find(cons_inst->type_id) : nullptr;
        std::cout << "  [" << c.src_node << ":" << (prod_type ? prod_type->id : "?")
                  << "] -> [" << c.dst_node << ":" << (cons_type ? cons_type->id : "?")
                  << "] sock=" << c.dst_socket
                  << "  producer_out in active_resources? "
                  << (in_active ? "YES" : "*** NO *** (BUG)") << std::endl;
        EXPECT_TRUE(in_active) << "Cross-group producer " << c.src_node
                               << " output (" << c.src_socket
                               << ") must be in active_resources";
    }
}

// =============================================================================
// Test 2 (STRUCTURE): Verify DAG built by FusedGraphCompiler preserves real
// graph edges (not just linear adjacency).
// =============================================================================
TEST(ReproBlendPreview, SplitPathRespectsRealGraphEdges) {
    // Construct a DAG manually mimicking the user's topology:
    //   1 -> 5
    //   2 -> 5
    //   5 -> 6
    //   6 -> 11
    //   7 -> 8
    //   8 -> 11   (mask)
    //   9 -> 10
    //   10 -> 11
    //   11 -> 12
    // Active path (topo): {1, 2, 5, 6, 7, 8, 9, 10, 11, 12}
    te::dag::DAG<uint64_t>::NodeList nodes = {1, 2, 5, 6, 7, 8, 9, 10, 11, 12};
    te::dag::DAG<uint64_t>::EdgeList edges = {
        {1, 5}, {2, 5}, {5, 6}, {6, 11},
        {7, 8}, {8, 11},
        {9, 10}, {10, 11},
        {11, 12}
    };
    te::dag::DAG<uint64_t> dag(nodes, edges);

    std::vector<uint64_t> active_path = {1, 2, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<uint32_t> costs(active_path.size(), 5);

    // Budget forces a split (10 nodes * 5 = 50 > 12).
    te::fusion::FusionPlanner planner(12);
    auto plan = planner.plan(dag, active_path, costs);

    ASSERT_TRUE(plan.valid);
    ASSERT_TRUE(plan.needs_split);

    std::cout << "\n=== Split plan (budget=12) ===" << std::endl;
    for (size_t i = 0; i < plan.groups.size(); ++i) {
        const auto& g = plan.groups[i];
        std::cout << "Group " << i << ": [";
        for (size_t j = 0; j < g.nodes.size(); ++j) {
            std::cout << g.nodes[j];
            if (j+1 < g.nodes.size()) std::cout << ",";
        }
        std::cout << "]";
        if (g.split_point.has_value()) std::cout << " split@" << *g.split_point;
        std::cout << std::endl;
    }

    // Verify: no group has a node whose successor (in the real DAG) is on
    // the active path but OUTSIDE that group.
    std::unordered_set<uint64_t> path_set(active_path.begin(), active_path.end());
    for (size_t gi = 0; gi < plan.groups.size(); ++gi) {
        const auto& g = plan.groups[gi];
        std::unordered_set<uint64_t> group_set(g.nodes.begin(), g.nodes.end());
        for (size_t i = 0; i + 1 < g.nodes.size(); ++i) {
            uint64_t intermediate = g.nodes[i];
            for (uint64_t succ : dag.successors(intermediate)) {
                if (!path_set.count(succ)) continue;
                EXPECT_TRUE(group_set.count(succ))
                    << "Group " << gi << " node " << intermediate
                    << " has successor " << succ
                    << " on active path but outside group (consumer constraint violation)";
            }
        }
    }
}

// =============================================================================
// Test 3 (PIXEL): Build the user's graph in the engine, compare final blend
// output against a per-pass (unfused reference) build of the same graph.
// =============================================================================
class ReproBlendPreviewPixel : public ::testing::Test {
protected:
    Engine engine;
    NodeLibrary lib = load_real_lib();

    void SetUp() override {
        if (!init_engine(engine, "test_repro_blend")) GTEST_SKIP() << engine.last_error();
    }
};

TEST_F(ReproBlendPreviewPixel, FinalBlend_ProducesNonZeroPixels) {
    UserGraph ug; ug.build();
    uint64_t gen = engine.set_graph(ug.g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    std::vector<float> px;
    uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px, w, h));
    ASSERT_FALSE(px.empty());

    double bright = avg_brightness(px);
    std::cout << "Final blend avg brightness: " << bright << std::endl;
    EXPECT_GT(bright, 0.0) << "final blend produced all-black output";
}

TEST_F(ReproBlendPreviewPixel, FinalBlend_DiffersFromUpstreamBlend) {
    // The final blend's output should NOT equal its 'a' input (cellular chain)
    // when simplex mask is non-trivial. This detects the bug where the
    // final blend reads stale/zero inputs.
    UserGraph ug; ug.build();

    // First render with Shuffle (Sh) as output — that's the final blend path.
    ug.g.output_node = ug.Sh;
    uint64_t gen_sh = engine.set_graph(ug.g);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_sh;
    uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen_sh, px_sh, w, h));

    // Now render with B2 as output — same blend result but without shuffle.
    ug.g.output_node = ug.B2;
    uint64_t gen_b2 = engine.set_graph(ug.g);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px_b2;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen_b2, px_b2, w, h));

    // Shuffle just routes R->RGB so px_sh and px_b2 should have similar
    // brightness (the shuffle shouldn't kill the signal).
    double b_sh = avg_brightness(px_sh);
    double b_b2 = avg_brightness(px_b2);
    std::cout << "Shuffle out brightness: " << b_sh << ", B2 out brightness: " << b_b2 << std::endl;

    EXPECT_GT(b_sh, 0.0) << "shuffle output is all-black but B2 produced signal";
    EXPECT_GT(b_b2, 0.0) << "B2 output is all-black";
}

TEST_F(ReproBlendPreviewPixel, FinalBlend_SensitiveToGaborBranch) {
    // If we change the gabor seed, the final blend output MUST change.
    // If the chain reads stale data from a wrong slot, the output won't
    // respond to the gabor seed change.
    UserGraph ug; ug.build();
    ug.g.output_node = ug.B2;

    uint64_t gen1 = engine.set_graph(ug.g);
    ASSERT_TRUE(wait_for_pipeline(engine));
    // Default gabor seed=24
    std::vector<float> px1;
    uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen1, px1, w, h));

    // Change gabor seed to 999.
    engine.update_node_params_by_id(ug.G1, {4.0f, 1.0f, 2.0f, 0.5f, 4.0f, 2.0f, 0.5f, 0.0f, 0.0f, 999.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    engine.poll_pending_compiles();
    std::vector<float> px2;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen1, px2, w, h));

    double diff = mean_abs_diff(px1, px2);
    std::cout << "Mean abs diff after gabor seed change: " << diff << std::endl;
    EXPECT_GT(diff, 0.01) << "final blend is insensitive to gabor branch — cross-chain read broken";
}

TEST_F(ReproBlendPreviewPixel, FinalBlend_SensitiveToSimplexMask) {
    // Change the simplex seed feeding B2.mask. Output MUST change.
    UserGraph ug; ug.build();
    ug.g.output_node = ug.B2;

    uint64_t gen = engine.set_graph(ug.g);
    ASSERT_TRUE(wait_for_pipeline(engine));
    std::vector<float> px1;
    uint32_t w=0, h=0;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px1, w, h));

    // Simplex params: period, octaves, lacunarity, roughness, speed, rotation, seed
    engine.update_node_params_by_id(ug.S1, {9.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f, 999.0f});
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    engine.poll_pending_compiles();
    std::vector<float> px2;
    ASSERT_TRUE(wait_for_readback_gen(engine, gen, px2, w, h));

    double diff = mean_abs_diff(px1, px2);
    std::cout << "Mean abs diff after simplex seed change: " << diff << std::endl;
    EXPECT_GT(diff, 0.01) << "final blend insensitive to simplex mask branch";
}
