#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/NodeLibrary.hpp"

using namespace te;

namespace {

NodeType make_type(const std::string& id,
                   uint32_t n_inputs,
                   uint32_t n_outputs,
                   uint32_t n_params = 0,
                   uint32_t n_socket_params = 0) {
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
    for (uint32_t i = 0; i < n_params; ++i) {
        NodeParam p;
        p.name = "p" + std::to_string(i);
        p.as_socket = (i < n_socket_params);
        t.params.push_back(p);
    }
    return t;
}

NodeLibrary make_library() {
    NodeLibrary lib;
    lib.add_public(make_type("source",      0, 1));
    lib.add_public(make_type("passthrough", 1, 1));
    lib.add_public(make_type("blend",       2, 1));
    lib.add_public(make_type("with_socket_param", 1, 1, 2, 1));
    return lib;
}

} // namespace

TEST(GraphValidation, RejectsEmptyGraph) {
    auto lib = make_library();
    Graph g;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("no nodes"), std::string::npos);
}

TEST(GraphValidation, AcceptsSingleSourceNode) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.output_node = 1;
    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    EXPECT_EQ(r.ir.nodes.size(), 1u);
    EXPECT_EQ(r.ir.eval_order.size(), 1u);
    EXPECT_EQ(r.ir.eval_order[0], 1u);
}

TEST(GraphValidation, RejectsUnknownNodeType) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "this_type_does_not_exist"});
    g.output_node = 1;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("Unknown node type"), std::string::npos);
}

TEST(GraphValidation, RejectsDuplicateNodeIds) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({1, "source"});
    g.output_node = 1;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("Duplicate"), std::string::npos);
}

TEST(GraphValidation, RejectsMissingOutputNode) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.output_node = 999;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("Output node"), std::string::npos);
}

TEST(GraphValidation, RejectsConnectionToMissingSourceNode) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "passthrough"});
    g.connections.push_back({999, 0, 1, 0});
    g.output_node = 1;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("source node"), std::string::npos);
}

TEST(GraphValidation, RejectsConnectionToMissingDestNode) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.connections.push_back({1, 0, 999, 0});
    g.output_node = 1;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("destination node"), std::string::npos);
}

TEST(GraphValidation, RejectsOutOfRangeSourceSocket) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough"});
    g.connections.push_back({1, 5, 2, 0});
    g.output_node = 2;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("source socket"), std::string::npos);
}

TEST(GraphValidation, RejectsOutOfRangeDestSocket) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough"});
    g.connections.push_back({1, 0, 2, 7});
    g.output_node = 2;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("destination socket"), std::string::npos);
}

TEST(GraphValidation, AcceptsConnectionIntoSocketDrivenParam) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "with_socket_param"});
    g.connections.push_back({1, 0, 2, 1});
    g.output_node = 2;
    auto r = validate_graph(g, lib);
    EXPECT_TRUE(r.success) << r.error;
}

TEST(GraphValidation, RejectsCycles) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "passthrough"});
    g.nodes.push_back({2, "passthrough"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 1, 0});
    g.output_node = 2;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("cycle"), std::string::npos);
}

TEST(GraphValidation, RejectsSelfLoop) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "passthrough"});
    g.connections.push_back({1, 0, 1, 0});
    g.output_node = 1;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
}

TEST(GraphValidation, EvalOrderIsTopological) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({10, "source"});
    g.nodes.push_back({20, "source"});
    g.nodes.push_back({30, "passthrough"});
    g.nodes.push_back({40, "blend"});
    g.connections.push_back({10, 0, 30, 0});
    g.connections.push_back({30, 0, 40, 0});
    g.connections.push_back({20, 0, 40, 1});
    g.output_node = 40;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto idx_of = [&](NodeId id) {
        for (size_t i = 0; i < r.ir.eval_order.size(); ++i)
            if (r.ir.eval_order[i] == id) return (int)i;
        return -1;
    };
    EXPECT_LT(idx_of(10), idx_of(30));
    EXPECT_LT(idx_of(30), idx_of(40));
    EXPECT_LT(idx_of(20), idx_of(40));
}

TEST(GraphValidation, PrunesUnreachableNodes) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough"});
    g.nodes.push_back({99, "source"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    EXPECT_EQ(r.ir.nodes.size(), 2u);
    EXPECT_NE(r.ir.find(1), nullptr);
    EXPECT_NE(r.ir.find(2), nullptr);
    EXPECT_EQ(r.ir.find(99), nullptr);
}

TEST(GraphValidation, NodeIndexIsConsistent) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success);
    for (auto& vn : r.ir.nodes) {
        auto it = r.ir.node_index.find(vn.id);
        ASSERT_NE(it, r.ir.node_index.end());
        EXPECT_EQ(r.ir.nodes[it->second].id, vn.id);
    }
}

TEST(GraphValidation, EvalOrderIsDeterministic) {
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({3, "source"});
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({3, 0, 2, 0});
    g.connections.push_back({1, 0, 2, 1});
    g.output_node = 2;

    auto r1 = validate_graph(g, lib);
    auto r2 = validate_graph(g, lib);
    ASSERT_TRUE(r1.success);
    ASSERT_TRUE(r2.success);
    EXPECT_EQ(r1.ir.eval_order, r2.ir.eval_order);
}

TEST(GraphValidation, RejectsLongCycle) {
    // A(1) -> B(2) -> C(3) -> D(4) -> A(1). Only D is a candidate output.
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "passthrough"});
    g.nodes.push_back({2, "passthrough"});
    g.nodes.push_back({3, "passthrough"});
    g.nodes.push_back({4, "passthrough"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({4, 0, 1, 0});
    g.output_node = 4;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    EXPECT_NE(r.error.find("cycle"), std::string::npos);
}

TEST(GraphValidation, AcceptsDiamondTopology) {
    // A -> B, A -> C, B -> D, C -> D. D is the output. No cycle.
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough"});
    g.nodes.push_back({3, "passthrough"});
    g.nodes.push_back({4, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 4, 0});
    g.connections.push_back({3, 0, 4, 1});
    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    EXPECT_EQ(r.ir.nodes.size(), 4u);
    EXPECT_EQ(r.ir.eval_order.size(), 4u);

    // Verify topological order: 1 < {2,3} < 4
    auto idx_of = [&](NodeId id) {
        for (size_t i = 0; i < r.ir.eval_order.size(); ++i)
            if (r.ir.eval_order[i] == id) return (int)i;
        return -1;
    };
    EXPECT_LT(idx_of(1), idx_of(2));
    EXPECT_LT(idx_of(1), idx_of(3));
    EXPECT_LT(idx_of(2), idx_of(4));
    EXPECT_LT(idx_of(3), idx_of(4));
}

TEST(GraphValidation, DebugNamePrefersUserValue) {
    // If NodeInstance::debug_name is set, ValidatedNode::debug_name should
    // match it verbatim. (Phase 1d: stable debug names from Python.)
    auto lib = make_library();
    Graph g;
    NodeInstance n;
    n.id = 1; n.type_id = "source"; n.debug_name = "TerrainBase";
    g.nodes.push_back(n);
    g.output_node = 1;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    const auto* vn = r.ir.find(1);
    ASSERT_NE(vn, nullptr);
    EXPECT_EQ(vn->debug_name, "TerrainBase");
}

TEST(GraphValidation, DebugNameFallsBackToTypeId) {
    // Empty debug_name should fall back to the auto-generated "{type}_{id}"
    // form so logs/error messages are still meaningful.
    auto lib = make_library();
    Graph g;
    NodeInstance n;
    n.id = 7; n.type_id = "source";  // debug_name left empty
    g.nodes.push_back(n);
    g.output_node = 7;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    const auto* vn = r.ir.find(7);
    ASSERT_NE(vn, nullptr);
    EXPECT_EQ(vn->debug_name, "source_7");
}

// ===========================================================================
// Phase 1c: muted / bypassed semantics
// ===========================================================================
//   muted    — node is removed from the active subgraph; downstream reads
//              this node's input[0] source instead. Implemented in
//              GraphIR::validate_graph as a connection rewire pass.
//   bypassed — node is retained in the active subgraph with a flag; the
//              compiler emits a clear-to-zero dispatch (executor-level
//              behavior, plumbed through ComputePass::bypassed).
// ===========================================================================

TEST(GraphValidation, MutedNodeIsRewiredAndSkippedInIR) {
    // A(1) -> M(2, muted) -> B(3) -> output.  B is `blend` (2 inputs), so
    // we route A into B's second input as well to keep the graph valid.
    // After validate: IR contains 1 and 3; connection 1->3 (rewired from
    // M->B) exists; node 2 is excluded entirely.
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough", ChannelFormat::RGBA, "", true,  false});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({1, 0, 3, 1});  // A also feeds B's 2nd input
    g.output_node = 3;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    EXPECT_EQ(r.ir.find(2), nullptr) << "muted node must not be in IR";
    ASSERT_NE(r.ir.find(1), nullptr);
    ASSERT_NE(r.ir.find(3), nullptr);
    EXPECT_EQ(r.ir.find(1)->muted,    false);
    EXPECT_EQ(r.ir.find(3)->muted,    false);
    EXPECT_EQ(r.ir.find(1)->bypassed, false);
    EXPECT_EQ(r.ir.find(3)->bypassed, false);

    // Verify the rewire: 1->3 connection exists, 2->3 does not.
    bool saw_1_to_3 = false, saw_2_to_3 = false;
    for (auto& c : r.ir.connections) {
        if (c.src_node == 1 && c.dst_node == 3) saw_1_to_3 = true;
        if (c.src_node == 2 && c.dst_node == 3) saw_2_to_3 = true;
    }
    EXPECT_TRUE(saw_1_to_3)  << "rewire should produce A->B connection";
    EXPECT_FALSE(saw_2_to_3) << "original M->B connection must be gone";
}

TEST(GraphValidation, BypassedNodeIsRetainedInIR) {
    // A(1) -> B(2, bypassed) -> output.
    // After validate: both nodes are present; B's bypassed flag is set.
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough", ChannelFormat::RGBA, "", false, true});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    EXPECT_NE(r.ir.find(1), nullptr);
    EXPECT_NE(r.ir.find(2), nullptr);
    EXPECT_EQ(r.ir.find(1)->bypassed, false);
    EXPECT_EQ(r.ir.find(2)->bypassed, true);
}

TEST(GraphValidation, MutedChainRewiresToFirstNonMutedSource) {
    // A(1) -> M1(2, muted) -> M2(3, muted) -> B(4) -> output.
    // After validate: IR has 1 and 4; connection 1->4 exists.
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough", ChannelFormat::RGBA, "", true,  false});
    g.nodes.push_back({3, "passthrough", ChannelFormat::RGBA, "", true,  false});
    g.nodes.push_back({4, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({1, 0, 4, 1});  // B's second input
    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    EXPECT_EQ(r.ir.find(2), nullptr);
    EXPECT_EQ(r.ir.find(3), nullptr);
    ASSERT_NE(r.ir.find(1), nullptr);
    ASSERT_NE(r.ir.find(4), nullptr);

    bool saw_1_to_4 = false, saw_2_to_4 = false, saw_3_to_4 = false;
    for (auto& c : r.ir.connections) {
        if (c.src_node == 1 && c.dst_node == 4) saw_1_to_4 = true;
        if (c.src_node == 2 && c.dst_node == 4) saw_2_to_4 = true;
        if (c.src_node == 3 && c.dst_node == 4) saw_3_to_4 = true;
    }
    EXPECT_TRUE(saw_1_to_4)  << "rewire should chain past muted nodes to A";
    EXPECT_FALSE(saw_2_to_4);
    EXPECT_FALSE(saw_3_to_4);
}

TEST(GraphValidation, MutedWithoutUpstreamSeveresConnection) {
    // M(1, muted, source) -> B(2) -> output. M has no input[0].
    // After validate: IR has only B; the M->B connection is severed
    // (dropped), leaving B's input[0] unsatisfied.
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source",      ChannelFormat::RGBA, "", true,  false});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 2, 1});  // A also feeds B's 2nd input
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    EXPECT_EQ(r.ir.find(1), nullptr);
    ASSERT_NE(r.ir.find(2), nullptr);

    // The severed 1->2 connection should be absent (src_node=0 sentinel
    // was filtered out).
    bool saw_severed = false;
    for (auto& c : r.ir.connections) {
        if (c.src_node == 0) saw_severed = true;
    }
    EXPECT_FALSE(saw_severed) << "severed connection should be dropped";
}

TEST(GraphValidation, BypassedFlagPropagatesToComputePass) {
    // A(1) -> B(2, bypassed) -> output.
    // After compile: pass for B has bypassed=true; pass for A does not.
    auto lib = make_library();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "passthrough", ChannelFormat::RGBA, "", false, true});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto comp = GraphCompiler::compile(r.ir, lib);
    ASSERT_TRUE(comp.success) << comp.error;
    ASSERT_EQ(comp.pass_plan.passes.size(), 2u);

    const ComputePass* pass_a = nullptr;
    const ComputePass* pass_b = nullptr;
    for (auto& p : comp.pass_plan.passes) {
        if (p.node_id == 1) pass_a = &p;
        if (p.node_id == 2) pass_b = &p;
    }
    ASSERT_NE(pass_a, nullptr);
    ASSERT_NE(pass_b, nullptr);
    EXPECT_FALSE(pass_a->bypassed);
    EXPECT_TRUE (pass_b->bypassed);
}
