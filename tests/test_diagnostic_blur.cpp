#include <gtest/gtest.h>
#include "test_helpers.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/FusionGroup.hpp"
#include "engine/graphfusion/FusionGroupEmitter.hpp"
#include "engine/graphfusion/FusedGroupCompiler.hpp"
#include "engine/Engine.hpp"
#include <cstdlib>

using namespace te;

namespace {

// ============================================================================
// Helper: build a NodeLibrary that matches the Blender addon's node types.
// Socket order must match the actual .node.json files for correct external
// input slot assignment.
// ============================================================================

NodeLibrary lib_for_blender_graphs() {
    NodeLibrary lib;
    // perlin: 0 inputs, 1 output, 0 params
    lib.add_public(make_type("perlin", 0, 1, 1, {},
        "vec4 node_perlin(vec2 uv) { return vec4(0.5); }", 0));
    // levels: 1 Vec4 input, 1 output, 25 params
    lib.add_public(make_type("levels", 1, 1, 1, {SocketType::Vec4},
        "vec4 node_levels(vec2 uv, vec4 color) { return color; }", 25));
    // blur: 1 Sampler2D input, 1 output, 1 param, 2 passes
    lib.add_public(make_type("blur", 1, 1, 2, {SocketType::Sampler2D},
        "vec4 node_blur(vec2 uv, TSTexture tex) { return vec4(0.0); }", 1));
    // worley: 0 inputs, 1 output, 7 params
    lib.add_public(make_type("worley", 0, 1, 1, {},
        "vec4 node_worley(vec2 uv) { return vec4(0.5); }", 7));
    // blend: {Float mask, Vec4 A, Vec4 B}, 1 output, 1 param (mode)
    lib.add_public(make_type("blend", 3, 1, 1, {SocketType::Float, SocketType::Vec4, SocketType::Vec4},
        "vec4 node_blend(vec2 uv, float mask, vec4 a, vec4 b) { return mix(a, b, mask); }", 1));
    return lib;
}

// ============================================================================
// Helper: dump all fusion-layer info for a given graph to stdout.
// ============================================================================

struct GraphInfo {
    Graph g;
    GraphIR ir;
    fusion::FusionContext ctx;
};

GraphInfo build_info(const Graph& graph, const NodeLibrary& lib) {
    GraphInfo info;
    info.g = graph;
    auto r = validate_graph(info.g, lib);
    EXPECT_TRUE(r.success) << r.error;
    info.ir = std::move(r.ir);
    info.ctx = fusion::build_context(info.ir, lib);
    return info;
}

void print_node_name(std::ostream& os, NodeId id, const GraphIR& ir) {
    const auto* n = ir.find(id);
    os << (n ? n->debug_name : "?") << "(" << id << ")";
}

void dump_fusion_diagnostics(const std::string& label, const GraphInfo& info,
                             const NodeLibrary& lib) {
    const auto& ir = info.ir;
    const auto& ctx = info.ctx;

    std::cout << "\n====== " << label << " ======\n";

    // 1. GraphIR dump
    std::cout << "\n--- GraphIR: nodes ---\n";
    for (const auto& vn : ir.nodes) {
        std::cout << "  id=" << vn.id << " type=" << vn.type_id
                  << " name=\"" << vn.debug_name << "\"\n";
    }
    std::cout << "\n--- GraphIR: connections ---\n";
    for (const auto& c : ir.connections) {
        auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type.at(c.dst_node) : nullptr;
        std::string socket_type = "?";
        if (st && c.dst_socket < st->inputs.size()) {
            if (st->inputs[c.dst_socket].type == SocketType::Vec4) socket_type = "Vec4";
            else if (st->inputs[c.dst_socket].type == SocketType::Sampler2D) socket_type = "Sampler2D";
            else if (st->inputs[c.dst_socket].type == SocketType::Float) socket_type = "Float";
        }
        std::cout << "  ";
        print_node_name(std::cout, c.src_node, ir);
        std::cout << " socket" << c.src_socket << " -> ";
        print_node_name(std::cout, c.dst_node, ir);
        std::cout << " socket" << c.dst_socket << " [" << socket_type << "]\n";
    }
    std::cout << "\n--- GraphIR: eval_order ---\n";
    for (size_t i = 0; i < ir.eval_order.size(); ++i) {
        NodeId id = ir.eval_order[i];
        std::cout << "  [" << i << "] ";
        print_node_name(std::cout, id, ir);
        std::cout << "\n";
    }

    // 2. Expanded order
    auto expanded = fusion::expand_multipass(ir.eval_order, ctx);
    std::cout << "\n--- Expanded order ---\n";
    for (size_t i = 0; i < expanded.size(); ++i) {
        auto& e = expanded[i];
        std::cout << "  [" << i << "] ";
        print_node_name(std::cout, e.node_id, ir);
        std::cout << " pass=" << e.pass_index << "/" << e.pass_count << "\n";
    }

    // 3. Grouping decisions (is_connected per adjacent pair)
    std::cout << "\n--- Grouping decisions (adjacent in expanded order) ---\n";
    for (size_t i = 0; i + 1 < expanded.size(); ++i) {
        NodeId n1 = expanded[i].node_id;
        NodeId n2 = expanded[i + 1].node_id;
        bool same = (n1 == n2);
        auto ct = fusion::get_connection_type(n1, n2, ctx);
        std::string type_str = "none";
        if (ct == SocketType::Vec4) type_str = "Vec4";
        else if (ct == SocketType::Sampler2D) type_str = "Sampler2D";
        else if (ct == SocketType::Float) type_str = "Float";
        bool conn = fusion::is_connected(n1, n2, ctx);
        std::cout << "  ";
        print_node_name(std::cout, n1, ir);
        std::cout << " -> ";
        print_node_name(std::cout, n2, ir);
        std::cout << ": type=" << type_str
                  << " conn=" << (conn ? "true" : "false")
                  << " same_node=" << (same ? "true" : "false")
                  << (conn ? " [MERGE]" : same ? " [SAME_NODE]" : " [BREAK]") << "\n";
    }

    // 4. Groups before merge
    auto fused = fusion::group_nodes(ir, ctx);
    std::cout << "\n--- Groups BEFORE merge (" << fused.groups.size() << " groups) ---\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        auto& g = fused.groups[gi];
        std::cout << "  Group " << gi << ": pass=" << g.pass_index << "/" << g.pass_count << " nodes=[";
        for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
            if (ni) std::cout << ", ";
            print_node_name(std::cout, g.nodes[ni], ir);
        }
        std::cout << "]\n";
    }

    // 5. After split+merge
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    std::cout << "\n--- Groups AFTER merge (" << fused.groups.size() << " groups) ---\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        auto& g = fused.groups[gi];
        std::cout << "  Group " << gi << ": pass=" << g.pass_index << "/" << g.pass_count
                  << " intermediate_count=" << g.intermediate_count << " nodes=[";
        for (size_t ni = 0; ni < g.nodes.size(); ++ni) {
            if (ni) std::cout << ", ";
            print_node_name(std::cout, g.nodes[ni], ir);
        }
        std::cout << "]\n";
    }

    // 6. Cross-group connections (any connection where src and dst are in different groups)
    std::cout << "\n--- Cross-group connections ---\n";
    for (const auto& c : ir.connections) {
        int src_group = -1, dst_group = -1;
        for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
            for (NodeId n : fused.groups[gi].nodes) {
                if (n == c.src_node) src_group = (int)gi;
                if (n == c.dst_node) dst_group = (int)gi;
            }
        }
        if (src_group != dst_group) {
            auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type.at(c.dst_node) : nullptr;
            std::string socket_type = "?";
            if (st && c.dst_socket < st->inputs.size()) {
                if (st->inputs[c.dst_socket].type == SocketType::Vec4) socket_type = "Vec4";
                else if (st->inputs[c.dst_socket].type == SocketType::Sampler2D) socket_type = "Sampler2D";
            }
            std::cout << "  ";
            print_node_name(std::cout, c.src_node, ir);
            std::cout << "(g" << src_group << ") -> ";
            print_node_name(std::cout, c.dst_node, ir);
            std::cout << "(g" << dst_group << ") [" << socket_type << "]\n";
        }
    }

    // 7. External inputs
    fusion::compute_external_inputs(fused, ctx);
    std::cout << "\n--- External inputs per group ---\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        auto& g = fused.groups[gi];
        if (g.external_inputs.empty()) {
            std::cout << "  Group " << gi << ": (none)\n";
            continue;
        }
        std::cout << "  Group " << gi << ":\n";
        for (auto& ext : g.external_inputs) {
            std::cout << "    slot=" << ext.slot << " ";
            print_node_name(std::cout, ext.src_node, ir);
            std::cout << " socket" << ext.src_socket << " -> ";
            print_node_name(std::cout, ext.dst_node, ir);
            std::cout << " socket" << ext.dst_socket << "\n";
        }
    }

    // 8. Compiled groups (GLSL output node, external_inputs, param layout)
    auto compiled = fusion::compile_groups(fused, ir, ctx, lib);
    (void)compiled;
    std::cout << "\n--- Compiled groups ---\n";
    for (size_t gi = 0; gi < compiled.groups.size(); ++gi) {
        auto& cg = compiled.groups[gi];
        std::cout << "  CompiledGroup " << gi << ": ok=" << (cg.ok() ? "true" : "false")
                  << " output_node=" << cg.output_node << " glsl_len=" << cg.glsl.size()
                  << " param_base_slot=" << cg.param_base_slot
                  << " param_floats=" << cg.param_floats
                  << " pass=" << cg.pass_index << "/" << cg.pass_count
                  << " intermediate_count=" << cg.intermediate_count << "\n";
        // Print external inputs for this compiled group
        if (!cg.external_inputs.empty()) {
            for (auto& ext : cg.external_inputs) {
                std::cout << "    ext: ";
                print_node_name(std::cout, ext.src_node, ir);
                std::cout << " socket" << ext.src_socket << " -> ";
                print_node_name(std::cout, ext.dst_node, ir);
                std::cout << " socket" << ext.dst_socket << " slot=" << ext.slot << "\n";
            }
        }
        // Print the GLSL main() body
        auto main_pos = cg.glsl.find("void main()");
        if (main_pos != std::string::npos) {
            std::cout << "  GLSL (from main()):\n";
            size_t end_pos = cg.glsl.find('\n', main_pos);
            if (end_pos != std::string::npos) {
                // Print 200 chars after the main function header
                std::string snippet = cg.glsl.substr(main_pos, 400);
                std::cout << snippet << "\n...\n";
            }
        } else if (!cg.glsl.empty()) {
            std::cout << "  GLSL (first 300 chars): " << cg.glsl.substr(0, 300) << "...\n";
        }
    }
    std::cout.flush();
}

} // anonymous namespace

// ============================================================================
// TEST 1: Perlin → Levels → Blur (Blur active as output)
// Mirrors the Blender node graph exactly: a Perlin noise through Levels
// then into Blur. Blur is the active/preview node.
// ============================================================================

TEST(DiagnosticBlur, Test1_PerlinLevelsBlur) {
    auto lib = lib_for_blender_graphs();

    Graph g;
    // Node IDs match Blender's actual IDs from MCP debug
    g.nodes.push_back({100, "perlin", ChannelFormat::RGBA, "Perlin Noise"});
    g.nodes.push_back({101, "levels", ChannelFormat::RGBA, "Levels"});
    g.nodes.push_back({102, "blur",   ChannelFormat::RGBA, "Blur"});

    // Perlin(100) -> Levels(101) socket 0 (Vec4)
    g.connections.push_back({100, 0, 101, 0});
    // Levels(101) -> Blur(102) socket 0 (Sampler2D)
    g.connections.push_back({101, 0, 102, 0});

    g.output_node = 102; // Blur is active/preview

    auto info = build_info(g, lib);
    dump_fusion_diagnostics("Test 1: Perlin->Levels->Blur (Blur active)", info, lib);
}

// ============================================================================
// TEST 2: Perlin → Levels → Blur → Blend + Worley → Blend (Blend active)
// Full graph as in Blender. Blend is active. Worley feeds blend B socket.
// ============================================================================

TEST(DiagnosticBlur, Test2_FullGraphBlendActive) {
    auto lib = lib_for_blender_graphs();

    Graph g;
    // Node IDs match Blender's actual IDs
    g.nodes.push_back({100, "perlin", ChannelFormat::RGBA, "Perlin Noise"});
    g.nodes.push_back({101, "levels", ChannelFormat::RGBA, "Levels"});
    g.nodes.push_back({102, "blur",   ChannelFormat::RGBA, "Blur"});
    g.nodes.push_back({104, "worley", ChannelFormat::RGBA, "Worley (Cellular)"});
    g.nodes.push_back({103, "blend",  ChannelFormat::RGBA, "Blend"});

    // Perlin(100) -> Levels(101) socket 0 (Vec4)
    g.connections.push_back({100, 0, 101, 0});
    // Levels(101) -> Blur(102) socket 0 (Sampler2D)
    g.connections.push_back({101, 0, 102, 0});
    // Blur(102) -> Blend(103) socket 2 (Vec4, second input "B")
    g.connections.push_back({102, 0, 103, 2});
    // Worley(104) -> Blend(103) socket 1 (Vec4, first input "A")
    g.connections.push_back({104, 0, 103, 1});

    g.output_node = 103; // Blend is active/preview

    auto info = build_info(g, lib);
    dump_fusion_diagnostics("Test 2: Full Perlin->Levels->Blur->Blend + Worley->Blend (Blend active)", info, lib);
}

// ============================================================================
// TEST 3: Engine-layer diagnostic — same graphs but through the full
// Engine compile path. Requires Vulkan; skips if init fails.
// ============================================================================

class EngineGroupDumpTest : public ::testing::Test {
protected:
    Engine engine;

    void SetUp() override {
        bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, false,
                             "test_shader_cache",
                             std::string(TEXTURESYNTH_TEST_ASSET_DIR) + "/shader_assets/nodes",
                             std::string(TEXTURESYNTH_TEST_ASSET_DIR) + "/shader_assets/glsl");
        if (!ok) {
            GTEST_SKIP() << "Engine::init failed (no Vulkan?)";
        }
    }

    void TearDown() override {
        engine.shutdown();
    }
};

TEST_F(EngineGroupDumpTest, Test1_PerlinLevelsBlur_EnginePath) {
    auto lib = lib_for_blender_graphs();
    Graph g;
    g.nodes.push_back({100, "perlin", ChannelFormat::RGBA, "Perlin Noise"});
    g.nodes.push_back({101, "levels", ChannelFormat::RGBA, "Levels"});
    g.nodes.push_back({102, "blur",   ChannelFormat::RGBA, "Blur"});
    g.connections.push_back({100, 0, 101, 0});
    g.connections.push_back({101, 0, 102, 0});
    g.output_node = 102;

    // GraphIR first for fusion analysis
    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    auto ir = std::move(r.ir);
    auto ctx = fusion::build_context(ir, lib);
    // Print the fusion diagnostics for reference
    dump_fusion_diagnostics("Test 1 (Engine path): Perlin->Levels->Blur", {g, ir, ctx}, lib);

    // Now run through the Engine
    engine.set_resolution(128, 128);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << "set_graph failed";

    // Wait for async compile
    for (int i = 0; i < 100; ++i) {
        engine.poll_pending_compiles();
        if (engine.is_generation_ready(gen)) break;
    }
    ASSERT_TRUE(engine.is_generation_ready(gen)) << "Engine compile did not ready within poll limit";

    // Dump group exec info
    engine.debug_dump_groups();
}

TEST_F(EngineGroupDumpTest, Test2_FullGraphBlendActive_EnginePath) {
    auto lib = lib_for_blender_graphs();
    Graph g;
    g.nodes.push_back({100, "perlin", ChannelFormat::RGBA, "Perlin Noise"});
    g.nodes.push_back({101, "levels", ChannelFormat::RGBA, "Levels"});
    g.nodes.push_back({102, "blur",   ChannelFormat::RGBA, "Blur"});
    g.nodes.push_back({104, "worley", ChannelFormat::RGBA, "Worley (Cellular)"});
    g.nodes.push_back({103, "blend",  ChannelFormat::RGBA, "Blend"});
    g.connections.push_back({100, 0, 101, 0});
    g.connections.push_back({101, 0, 102, 0});
    g.connections.push_back({102, 0, 103, 2});
    g.connections.push_back({104, 0, 103, 1});
    g.output_node = 103;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    auto ir = std::move(r.ir);
    auto ctx = fusion::build_context(ir, lib);
    dump_fusion_diagnostics("Test 2 (Engine path): Full graph Blend active", {g, ir, ctx}, lib);

    engine.set_resolution(128, 128);
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << "set_graph failed";

    for (int i = 0; i < 100; ++i) {
        engine.poll_pending_compiles();
        if (engine.is_generation_ready(gen)) break;
    }
    ASSERT_TRUE(engine.is_generation_ready(gen)) << "Engine compile did not ready within poll limit";

    engine.debug_dump_groups();
}
