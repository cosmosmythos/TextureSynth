#include <algorithm>
#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"

using namespace te;

namespace {

NodeType make_type(const std::string& id,
                   uint32_t n_in,
                   uint32_t n_out,
                   uint32_t n_params = 0) {
    NodeType t;
    t.id = id;
    t.display_name = id;
    t.pass_kind = PassKind::Compute;
    for (uint32_t i = 0; i < n_in; ++i) {
        Socket s; s.name = "in" + std::to_string(i); s.type = SocketType::Vec4;
        t.inputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        Socket s; s.name = "out" + std::to_string(i); s.type = SocketType::Vec4;
        t.outputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_params; ++i) {
        NodeParam p; p.name = "p" + std::to_string(i); t.params.push_back(p);
    }
    t.glsl_function = "vec4 node_" + id + "(vec2 uv";
    for (uint32_t i = 0; i < n_in; ++i)
        t.glsl_function += ", vec4 in" + std::to_string(i);
    for (uint32_t i = 0; i < n_params; ++i)
        t.glsl_function += ", float p" + std::to_string(i);
    t.glsl_function += ") { return vec4(0.0); }";
    return t;
}

NodeLibrary make_lib() {
    NodeLibrary lib;
    lib.add_public(make_type("source", 0, 1));
    lib.add_public(make_type("step",   1, 1));
    lib.add_public(make_type("step1p", 1, 1, 1));
    lib.add_public(make_type("blend",  2, 1));
    return lib;
}

} // namespace

TEST(FusedGraphCompiler, LinearChainProducesFusedChain) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto result = FusedGraphCompiler::compile(r.ir, lib, 3);
    ASSERT_TRUE(result.success) << result.error;

    // All 3 nodes should have ComputePass entries.
    EXPECT_EQ(result.pass_plan.passes.size(), 3u);

    // At least one chain should exist (linear chain fits in register budget).
    EXPECT_GE(result.pass_plan.chains.size(), 1u);

    // chain_index_of_pass should map all 3 nodes into the same chain.
    uint32_t chain_idx = result.pass_plan.chain_index_of_pass[0];
    EXPECT_NE(chain_idx, UINT32_MAX);
    for (uint32_t i = 1; i < 3; ++i)
        EXPECT_EQ(result.pass_plan.chain_index_of_pass[i], chain_idx);

    // Chain should contain all 3 nodes in topo order.
    const auto& chain = result.pass_plan.chains[chain_idx];
    EXPECT_EQ(chain.nodes.size(), 3u);
    EXPECT_EQ(chain.nodes[0], 1u); // source
    EXPECT_EQ(chain.nodes[1], 2u); // step
    EXPECT_EQ(chain.nodes[2], 3u); // step

    // Chain should have fused GLSL.
    EXPECT_FALSE(chain.glsl.empty());
    EXPECT_NE(chain.glsl.find("node_source"), std::string::npos);
    EXPECT_NE(chain.glsl.find("node_step"), std::string::npos);
    EXPECT_NE(chain.glsl.find("imageStore"), std::string::npos);

    // Variant key should be populated.
    EXPECT_FALSE(chain.variant_key.node_type_ids.empty());
    EXPECT_EQ(chain.variant_key.node_type_ids.size(), 3u);
}

TEST(FusedGraphCompiler, ExternalInputChain) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    // Blend node has 2 inputs but only socket 0 is connected.
    // Active path to node 3 includes source→step→blend.
    // Blend socket 1 is unconnected Vec4 — baked as vec4(0.0).
    auto result = FusedGraphCompiler::compile(r.ir, lib, 3);
    ASSERT_TRUE(result.success) << result.error;

    // At least one chain should exist.
    EXPECT_GE(result.pass_plan.chains.size(), 1u);

    // Find the chain containing node 3.
    for (auto& chain : result.pass_plan.chains) {
        bool has_blend = false;
        for (NodeId n : chain.nodes) {
            if (n == 3) { has_blend = true; break; }
        }
        if (!has_blend) continue;

        // Unconnected Vec4 baked as constant — no external input slot consumed.
        EXPECT_TRUE(chain.external_socket_masks.empty() ||
                    std::all_of(chain.external_socket_masks.begin(),
                                chain.external_socket_masks.end(),
                                [](uint32_t m) { return m == 0; }));
        EXPECT_EQ(chain.glsl.find("texelFetch"), std::string::npos)
            << "unconnected Vec4 must not use texelFetch";
        break;
    }
}

TEST(FusedGraphCompiler, ParamsInChain) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step1p"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto result = FusedGraphCompiler::compile(r.ir, lib, 2);
    ASSERT_TRUE(result.success) << result.error;

    EXPECT_GE(result.pass_plan.chains.size(), 1u);
    for (auto& chain : result.pass_plan.chains) {
        if (chain.nodes.size() < 2) continue;
        // Chain should have param data.
        EXPECT_EQ(chain.param_offsets.size(), 2u);
        EXPECT_EQ(chain.param_global_slots.size(), 2u);
        EXPECT_GE(chain.total_params, 1u);
        // GLSL should have param references.
        EXPECT_NE(chain.glsl.find("node_params"), std::string::npos);
        break;
    }
}

TEST(FusedGraphCompiler, EmptyGraphReturnsError) {
    NodeLibrary lib;
    GraphIR ir;
    auto result = FusedGraphCompiler::compile(ir, lib, 0);
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.error.empty());
}

TEST(FusedGraphCompiler, ParamBaseSlotCorrect) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step1p"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto result = FusedGraphCompiler::compile(r.ir, lib, 2);
    ASSERT_TRUE(result.success) << result.error;

    // Node 1 (source) has 0 params → base slot 0.
    EXPECT_EQ(result.param_base_slot[1], 0);
    // Node 2 (step1p) has 1 param → base slot 0 + 0 = 0.
    EXPECT_EQ(result.param_base_slot[2], 0);

    // Total param floats = 0 (source) + 1 (step1p) = 1.
    EXPECT_EQ(result.total_param_floats, 1);
}

TEST(FusedGraphCompiler, FanOutProducesChain) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    g.nodes.push_back({4, "blend"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 4, 0});
    g.connections.push_back({3, 0, 4, 1});
    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    // Active path to node 4: source→step(2)→blend(4)
    // source→step(3) is NOT in the active path (step3 isn't ancestor of 4 via 2).
    // Actually wait: blend has both step2 and step3 as inputs. BFS from 4:
    // predecessors(4) = {2,3}, then predecessors(2) = {1}, predecessors(3) = {1}.
    // So active path = [1,2,3,4] — both branches are ancestors of 4.
    auto result = FusedGraphCompiler::compile(r.ir, lib, 4);
    ASSERT_TRUE(result.success) << result.error;

    EXPECT_GE(result.pass_plan.chains.size(), 1u);
    // Chain should contain at least source, step2, blend (and possibly step3).
    for (auto& chain : result.pass_plan.chains) {
        EXPECT_GE(chain.nodes.size(), 3u);
        // Should include source and blend.
        bool has_source = false, has_blend = false;
        for (NodeId n : chain.nodes) {
            if (n == 1) has_source = true;
            if (n == 4) has_blend = true;
        }
        EXPECT_TRUE(has_source);
        EXPECT_TRUE(has_blend);
    }
}

TEST(FusedGraphCompiler, SplitPathObeysConsumerConstraint) {
    // Construct a DAG manually:
    // Nodes: 0, 1, 2, 3
    // Edges: (0, 1), (1, 2), (3, 2)
    te::dag::DAG<uint64_t>::NodeList nodes = {0, 1, 2, 3};
    te::dag::DAG<uint64_t>::EdgeList edges = {
        {0, 1}, {1, 2}, {3, 2}
    };
    te::dag::DAG<uint64_t> dag(nodes, edges);

    // We pass a topological active path: {0, 3, 1, 2}
    std::vector<uint64_t> active_path = {0, 3, 1, 2};
    std::vector<uint32_t> costs = {5, 5, 5, 5};

    // Set budget to 12.
    te::fusion::FusionPlanner planner(12);
    auto plan = planner.plan(dag, active_path, costs);

    ASSERT_TRUE(plan.valid);
    ASSERT_TRUE(plan.needs_split);

    // Since {0, 3} is invalid due to successor 1 of 0 being outside the group,
    // the planner must split at 0, making {0} a group.
    // Next, Group 1 candidate {3, 1} is invalid since successor 2 of 3 is outside {3, 1}.
    // So it must split at 3.
    // Lastly, Group 2 candidate {1, 2} has successor 2 of 1 inside the group. Valid!
    //
    // So the groups must be:
    // Group 0: {0}
    // Group 1: {3}
    // Group 2: {1, 2}

    ASSERT_EQ(plan.groups.size(), 3u);
    EXPECT_EQ(plan.groups[0].nodes, std::vector<uint64_t>({0}));
    EXPECT_EQ(plan.groups[1].nodes, std::vector<uint64_t>({3}));
    EXPECT_EQ(plan.groups[2].nodes, std::vector<uint64_t>({1, 2}));
}

