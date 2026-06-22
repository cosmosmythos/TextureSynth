// Tests for SD-style depth inheritance (resolve_node_depths).
// Pure C++ — no Vulkan, no Engine. Tests the GraphIR resolution logic directly.

#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"

using namespace te;

namespace {

NodeType make_type(const std::string& id,
                   uint32_t n_inputs = 0,
                   uint32_t n_outputs = 1) {
    NodeType t;
    t.id = id;
    t.display_name = id;
    for (uint32_t i = 0; i < n_inputs; ++i) {
        Socket s; s.name = "in" + std::to_string(i); s.type = SocketType::Vec4;
        t.inputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_outputs; ++i) {
        Socket s; s.name = "out" + std::to_string(i); s.type = SocketType::Vec4;
        t.outputs.push_back(s);
    }
    return t;
}

NodeLibrary make_lib() {
    NodeLibrary lib;
    lib.add_public(make_type("source", 0, 1));
    lib.add_public(make_type("pass", 1, 1));
    lib.add_public(make_type("blend", 2, 1));
    return lib;
}

// Helper: validate a graph and stamp graph_default_depth, then resolve depths.
GraphIR resolve(Graph& g, BitDepth default_depth) {
    auto lib = make_lib();
    auto result = validate_graph(g, lib);
    EXPECT_TRUE(result.success) << result.error;
    result.ir.graph_default_depth = default_depth;
    resolve_node_depths(result.ir);
    return std::move(result.ir);
}

const ValidatedNode* find_node(const GraphIR& ir, NodeId id) {
    return ir.find(id);
}

} // namespace

// ── Auto mode: inherits graph_default_depth ──────────────────────

TEST(DepthResolution, AutoInheritsGraphDefaultF16) {
    Graph g;
    g.nodes.push_back({1, "source"});
    g.output_node = 1;

    auto ir = resolve(g, BitDepth::F16);
    auto* n = find_node(ir, 1);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->resolved_depth, BitDepth::F16);
}

TEST(DepthResolution, AutoInheritsGraphDefaultF8) {
    Graph g;
    g.nodes.push_back({1, "source"});
    g.output_node = 1;

    auto ir = resolve(g, BitDepth::F8);
    auto* n = find_node(ir, 1);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->resolved_depth, BitDepth::F8);
}

TEST(DepthResolution, AutoInheritsGraphDefaultF32) {
    Graph g;
    g.nodes.push_back({1, "source"});
    g.output_node = 1;

    auto ir = resolve(g, BitDepth::F32);
    auto* n = find_node(ir, 1);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->resolved_depth, BitDepth::F32);
}

// ── Absolute mode: uses node's absolute_depth directly ───────────

TEST(DepthResolution, AbsoluteIgnoresGraphDefault) {
    Graph g;
    NodeInstance ni;
    ni.id = 1;
    ni.type_id = "source";
    ni.depth_mode = DepthMode::Absolute;
    ni.absolute_depth = BitDepth::F32;
    g.nodes.push_back(ni);
    g.output_node = 1;

    auto ir = resolve(g, BitDepth::F8);  // graph default is F8, but Absolute overrides
    auto* n = find_node(ir, 1);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->resolved_depth, BitDepth::F32);
}

TEST(DepthResolution, AbsoluteF8OnF32Graph) {
    Graph g;
    NodeInstance ni;
    ni.id = 1;
    ni.type_id = "source";
    ni.depth_mode = DepthMode::Absolute;
    ni.absolute_depth = BitDepth::F8;
    g.nodes.push_back(ni);
    g.output_node = 1;

    auto ir = resolve(g, BitDepth::F32);
    auto* n = find_node(ir, 1);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->resolved_depth, BitDepth::F8);
}

// ── MatchInput mode: inherits upstream's resolved_depth ──────────

TEST(DepthResolution, MatchInputInheritsFromUpstream) {
    Graph g;
    // source: Absolute F32
    NodeInstance src;
    src.id = 1;
    src.type_id = "source";
    src.depth_mode = DepthMode::Absolute;
    src.absolute_depth = BitDepth::F32;
    g.nodes.push_back(src);

    // pass: MatchInput — should pick up F32 from source
    NodeInstance p;
    p.id = 2;
    p.type_id = "pass";
    p.depth_mode = DepthMode::MatchInput;
    g.nodes.push_back(p);

    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto ir = resolve(g, BitDepth::F16);  // graph default is F16, but MatchInput picks F32
    auto* src_n = find_node(ir, 1);
    auto* pass_n = find_node(ir, 2);
    ASSERT_NE(src_n, nullptr);
    ASSERT_NE(pass_n, nullptr);
    EXPECT_EQ(src_n->resolved_depth, BitDepth::F32);
    EXPECT_EQ(pass_n->resolved_depth, BitDepth::F32);
}

TEST(DepthResolution, MatchInputFallsBackToGraphDefaultWhenNoUpstream) {
    Graph g;
    // pass with no input connection — MatchInput should fall back to graph default
    NodeInstance p;
    p.id = 1;
    p.type_id = "pass";
    p.depth_mode = DepthMode::MatchInput;
    g.nodes.push_back(p);
    g.output_node = 1;

    auto ir = resolve(g, BitDepth::F8);
    auto* n = find_node(ir, 1);
    ASSERT_NE(n, nullptr);
    EXPECT_EQ(n->resolved_depth, BitDepth::F8);
}

// ── Multi-node chain: depth propagates through chain ─────────────

TEST(DepthResolution, ChainPropagationAutoToMatchInput) {
    Graph g;
    // source: Auto (F16 graph default)
    NodeInstance src;
    src.id = 1;
    src.type_id = "source";
    src.depth_mode = DepthMode::Auto;
    g.nodes.push_back(src);

    // pass1: Absolute F8
    NodeInstance p1;
    p1.id = 2;
    p1.type_id = "pass";
    p1.depth_mode = DepthMode::Absolute;
    p1.absolute_depth = BitDepth::F8;
    g.nodes.push_back(p1);

    // pass2: MatchInput — should pick up F8 from pass1
    NodeInstance p2;
    p2.id = 3;
    p2.type_id = "pass";
    p2.depth_mode = DepthMode::MatchInput;
    g.nodes.push_back(p2);

    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    auto ir = resolve(g, BitDepth::F32);
    EXPECT_EQ(find_node(ir, 1)->resolved_depth, BitDepth::F32);  // Auto -> graph default
    EXPECT_EQ(find_node(ir, 2)->resolved_depth, BitDepth::F8);   // Absolute
    EXPECT_EQ(find_node(ir, 3)->resolved_depth, BitDepth::F8);   // MatchInput -> pass1's F8
}

TEST(DepthResolution, BlendNodePickFirstInputDepth) {
    Graph g;
    // source1: Absolute F32
    NodeInstance s1;
    s1.id = 1;
    s1.type_id = "source";
    s1.depth_mode = DepthMode::Absolute;
    s1.absolute_depth = BitDepth::F32;
    g.nodes.push_back(s1);

    // source2: Absolute F8
    NodeInstance s2;
    s2.id = 2;
    s2.type_id = "source";
    s2.depth_mode = DepthMode::Absolute;
    s2.absolute_depth = BitDepth::F8;
    g.nodes.push_back(s2);

    // blend: MatchInput — picks up from first connection found (input[0] = source1)
    NodeInstance bl;
    bl.id = 3;
    bl.type_id = "blend";
    bl.depth_mode = DepthMode::MatchInput;
    g.nodes.push_back(bl);

    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 3, 1});
    g.output_node = 3;

    auto ir = resolve(g, BitDepth::F16);
    EXPECT_EQ(find_node(ir, 1)->resolved_depth, BitDepth::F32);
    EXPECT_EQ(find_node(ir, 2)->resolved_depth, BitDepth::F8);
    // MatchInput on blend: iterates connections, first match is dst_node=3, src_node=1
    EXPECT_EQ(find_node(ir, 3)->resolved_depth, BitDepth::F32);
}

// ── Default: all nodes default to Auto/F16 ───────────────────────

TEST(DepthResolution, AllDefaultsAreF16) {
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "pass"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto ir = resolve(g, BitDepth::F16);
    EXPECT_EQ(find_node(ir, 1)->resolved_depth, BitDepth::F16);
    EXPECT_EQ(find_node(ir, 2)->resolved_depth, BitDepth::F16);
}

// ── Eval order matters: MatchInput reads already-resolved upstream ─

TEST(DepthResolution, EvalOrderMattersForMatchInput) {
    Graph g;
    // If eval_order is [1, 2, 3] (topo sort), MatchInput on node 3
    // should see node 2's resolved depth (F8), not node 2's absolute_depth.
    // They happen to be the same here, but the key point is that
    // resolve_node_depths walks eval_order, not raw node list.

    NodeInstance s;
    s.id = 1;
    s.type_id = "source";
    s.depth_mode = DepthMode::Absolute;
    s.absolute_depth = BitDepth::F32;
    g.nodes.push_back(s);

    NodeInstance p;
    p.id = 2;
    p.type_id = "pass";
    p.depth_mode = DepthMode::Absolute;
    p.absolute_depth = BitDepth::F8;
    g.nodes.push_back(p);

    NodeInstance end;
    end.id = 3;
    end.type_id = "pass";
    end.depth_mode = DepthMode::MatchInput;
    g.nodes.push_back(end);

    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    auto ir = resolve(g, BitDepth::F16);
    EXPECT_EQ(find_node(ir, 3)->resolved_depth, BitDepth::F8);
}

// ── Mixed depths across graph ────────────────────────────────────

TEST(DepthResolution, MixedDepthsAcrossGraph) {
    Graph g;
    NodeInstance s1;
    s1.id = 1;
    s1.type_id = "source";
    s1.depth_mode = DepthMode::Auto;  // F16 (graph default)
    g.nodes.push_back(s1);

    NodeInstance s2;
    s2.id = 2;
    s2.type_id = "source";
    s2.depth_mode = DepthMode::Absolute;
    s2.absolute_depth = BitDepth::F8;
    g.nodes.push_back(s2);

    NodeInstance s3;
    s3.id = 3;
    s3.type_id = "source";
    s3.depth_mode = DepthMode::Absolute;
    s3.absolute_depth = BitDepth::F32;
    g.nodes.push_back(s3);

    NodeInstance blend;
    blend.id = 4;
    blend.type_id = "blend";
    blend.depth_mode = DepthMode::Auto;  // F16 (graph default)
    g.nodes.push_back(blend);

    g.connections.push_back({1, 0, 4, 0});
    g.connections.push_back({2, 0, 4, 1});
    g.output_node = 4;

    auto ir = resolve(g, BitDepth::F16);
    EXPECT_EQ(find_node(ir, 1)->resolved_depth, BitDepth::F16);  // Auto
    EXPECT_EQ(find_node(ir, 2)->resolved_depth, BitDepth::F8);   // Absolute
    EXPECT_EQ(find_node(ir, 3)->resolved_depth, BitDepth::F32);  // Absolute (not connected to output)
    EXPECT_EQ(find_node(ir, 4)->resolved_depth, BitDepth::F16);  // Auto
}
