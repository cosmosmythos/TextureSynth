#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "test_assets.hpp"
#include <fstream>
#include <iostream>

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
} // namespace

class BlenderGraphDump : public ::testing::Test {
protected:
    NodeLibrary lib = load_lib();
};

TEST_F(BlenderGraphDump, DumpBlend001Graph) {
    // Mirror the exact Blender graph (13 nodes, 13 links).
    // IDs match Blender stable_id() values.
    Graph g;

    // Node IDs (from Blender introspection)
    const uint64_t PERLIN      = 3323707810362444727ULL;
    const uint64_t WORLEY      = 17786771187348095680ULL;
    const uint64_t INVERT      = 13466220939101946320ULL;
    const uint64_t BLUR        = 16517722687228889292ULL;
    const uint64_t WARP        = 11486196266130621948ULL;
    const uint64_t LEVELS      = 5086886035144592915ULL;
    const uint64_t WORLEY2     = 7059787924525040894ULL;
    const uint64_t LEVELS2     = 11003430421407682113ULL;
    const uint64_t WARP2       = 15798629895977455370ULL;
    const uint64_t LEVELS3     = 6347427700682145894ULL;
    const uint64_t BLEND       = 4115388978714529103ULL;
    const uint64_t PERLIN2     = 18270700326706430356ULL;
    const uint64_t BLEND2      = 12969497203451514664ULL;

    // Add nodes
    g.nodes.push_back({PERLIN,  "perlin"});
    g.nodes.push_back({WORLEY,  "worley"});
    g.nodes.push_back({INVERT,  "invert"});
    g.nodes.push_back({BLUR,    "blur"});
    g.nodes.push_back({WARP,    "warp"});
    g.nodes.push_back({LEVELS,  "levels"});
    g.nodes.push_back({WORLEY2, "worley"});
    g.nodes.push_back({LEVELS2, "levels"});
    g.nodes.push_back({WARP2,   "warp"});
    g.nodes.push_back({LEVELS3, "levels"});
    g.nodes.push_back({BLEND,   "blend"});
    g.nodes.push_back({PERLIN2, "perlin"});
    g.nodes.push_back({BLEND2,  "blend"});

    // Connections (from Blender introspection)
    // Worley -> Blur
    g.connections.push_back({WORLEY, 0, BLUR, 0});
    // Blur -> Invert.color (socket 1)
    g.connections.push_back({BLUR, 0, INVERT, 1});
    // Invert -> Warp.gradient (socket 1)
    g.connections.push_back({INVERT, 0, WARP, 1});
    // Levels -> Warp.image (socket 0)
    g.connections.push_back({LEVELS, 0, WARP, 0});
    // Worley2 -> Levels2
    g.connections.push_back({WORLEY2, 0, LEVELS2, 0});
    // Levels2 -> Warp2.image (socket 0)
    g.connections.push_back({LEVELS2, 0, WARP2, 0});
    // Perlin -> Warp2.gradient (socket 1)
    g.connections.push_back({PERLIN, 0, WARP2, 1});
    // Warp2 -> Levels3
    g.connections.push_back({WARP2, 0, LEVELS3, 0});
    // Perlin2 -> Levels
    g.connections.push_back({PERLIN2, 0, LEVELS, 0});
    // Levels3 -> Blend.a (socket 1)
    g.connections.push_back({LEVELS3, 0, BLEND, 1});
    // Warp -> Blend.b (socket 2)
    g.connections.push_back({WARP, 0, BLEND, 2});
    // Blend -> Blend2.a (socket 1)
    g.connections.push_back({BLEND, 0, BLEND2, 1});
    // Warp -> Blend2.b (socket 2)
    g.connections.push_back({WARP, 0, BLEND2, 2});

    g.output_node = BLEND2;

    // Compile
    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << "validate_graph failed: " << r.error;

    auto cr = FusedGraphCompiler::compile(r.ir, lib, g.output_node);
    ASSERT_TRUE(cr.success) << "compile failed: " << cr.error;

    // Dump chain info
    std::cout << "\n=== Chain Summary ===\n";
    std::cout << "Total chains: " << cr.pass_plan.chains.size() << "\n";
    for (size_t i = 0; i < cr.pass_plan.chains.size(); ++i) {
        const auto& ch = cr.pass_plan.chains[i];
        std::cout << "Chain " << i << ": nodes=[";
        for (size_t j = 0; j < ch.nodes.size(); ++j) {
            if (j > 0) std::cout << ", ";
            // Find node type
            auto* vn = r.ir.find(ch.nodes[j]);
            auto* type = vn ? lib.find(vn->type_id) : nullptr;
            std::string name = type ? type->id : "?";
            std::cout << ch.nodes[j] << "(" << name << ")";
        }
        std::cout << "] params=" << ch.total_params
                  << " glsl_bytes=" << ch.glsl.size()
                  << " sub_pass_count=" << ch.sub_pass_count << "\n";

        // Dump GLSL to file
        if (!ch.glsl.empty()) {
            std::string filename = "blender_chain_" + std::to_string(i) + ".glsl";
            std::ofstream f(filename);
            f << ch.glsl;
            f.close();
            std::cout << "  -> " << filename << "\n";
        }
        for (size_t sp = 0; sp < ch.sub_pass_glsl.size(); ++sp) {
            if (!ch.sub_pass_glsl[sp].empty()) {
                std::string filename = "blender_chain_" + std::to_string(i)
                                       + "_sub" + std::to_string(sp) + ".glsl";
                std::ofstream f(filename);
                f << ch.sub_pass_glsl[sp];
                f.close();
                std::cout << "  -> " << filename << " (sub-pass " << sp << ")\n";
            }
        }
    }

    // Dump per-pass info
    std::cout << "\n=== Pass Summary ===\n";
    for (size_t i = 0; i < cr.pass_plan.passes.size(); ++i) {
        const auto& p = cr.pass_plan.passes[i];
        auto* vn = r.ir.find(p.node_id);
        auto* type = vn ? lib.find(vn->type_id) : nullptr;
        std::string name = type ? type->id : "?";
        uint32_t chain_idx = (i < cr.pass_plan.chain_index_of_pass.size())
                            ? cr.pass_plan.chain_index_of_pass[i] : UINT32_MAX;
        std::cout << "Pass " << i << ": node=" << p.node_id << "(" << name << ")"
                  << " chain=" << (chain_idx == UINT32_MAX ? "none" : std::to_string(chain_idx))
                  << " glsl_bytes=" << p.shader_glsl.size() << "\n";
    }

    // Verify: Blend.001 should be in a chain
    bool found_blend2 = false;
    for (const auto& ch : cr.pass_plan.chains) {
        for (NodeId n : ch.nodes) {
            if (n == BLEND2) { found_blend2 = true; break; }
        }
    }
    EXPECT_TRUE(found_blend2) << "Blend.001 (id=" << BLEND2 << ") must be in a chain";

    // Dump alias coloring
    std::cout << "\n=== Alias Coloring ===\n";
    std::cout << "color_classes entries: " << cr.pass_plan.color_classes.size() << "\n";
    for (const auto& kv : cr.pass_plan.color_classes) {
        auto* vn = r.ir.find(kv.first.node_id);
        auto* type = vn ? lib.find(vn->type_id) : nullptr;
        std::string name = type ? type->id : "?";
        std::cout << "  node=" << kv.first.node_id << "(" << name << ")"
                  << "_out" << kv.first.output_index
                  << " -> color=" << kv.second << "\n";
    }

    // Dump lifetimes
    std::cout << "\n=== Lifetimes ===\n";
    for (const auto& kv : cr.pass_plan.lifetimes) {
        auto* vn = r.ir.find(kv.first.node_id);
        auto* type = vn ? lib.find(vn->type_id) : nullptr;
        std::string name = type ? type->id : "?";
        std::cout << "  node=" << kv.first.node_id << "(" << name << ")"
                  << "_out" << kv.first.output_index
                  << " first=" << kv.second.first_pass
                  << " last=" << kv.second.last_pass;
        if (kv.second.last_pass == UINT32_MAX) std::cout << " (PINNED)";
        std::cout << "\n";
    }

    // Dump active_resources
    std::cout << "\n=== Active Resources ===\n";
    for (const auto& rid : cr.pass_plan.active_resources) {
        auto* vn = r.ir.find(rid.node_id);
        auto* type = vn ? lib.find(vn->type_id) : nullptr;
        std::string name = type ? type->id : "?";
        auto cc = cr.pass_plan.color_classes.find(rid);
        uint32_t color = (cc != cr.pass_plan.color_classes.end()) ? cc->second : 0;
        std::cout << "  node=" << rid.node_id << "(" << name << ")"
                  << "_out" << rid.output_index
                  << " color=" << color << "\n";
    }
}
