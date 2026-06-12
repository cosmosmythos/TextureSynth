#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include <algorithm>

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
    return t;
}

NodeLibrary make_lib() {
    NodeLibrary lib;
    lib.add_public(make_type("source", 0, 1));
    lib.add_public(make_type("step",   1, 1));
    lib.add_public(make_type("blend",  2, 1));
    return lib;
}

} // namespace

TEST(ActivePathTracer, LinearChain) {
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

    auto path = ActivePathTracer::trace(r.ir, 3, lib);

    ASSERT_EQ(path.nodes.size(), 3u);
    EXPECT_EQ(path.nodes[0], 1u);
    EXPECT_EQ(path.nodes[1], 2u);
    EXPECT_EQ(path.nodes[2], 3u);
    EXPECT_TRUE(path.branch_points.empty());
    EXPECT_TRUE(path.merge_points.empty());
    EXPECT_GT(path.estimated_registers, 0u);
}

TEST(ActivePathTracer, ActiveNodeMiddleOfChain) {
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

    // Select node 2 (middle) as active
    auto path = ActivePathTracer::trace(r.ir, 2, lib);

    ASSERT_EQ(path.nodes.size(), 2u);
    EXPECT_EQ(path.nodes[0], 1u);
    EXPECT_EQ(path.nodes[1], 2u);
    // Node 3 should NOT be in the path
    EXPECT_TRUE(std::find(path.nodes.begin(), path.nodes.end(), 3) == path.nodes.end());
}

TEST(ActivePathTracer, BranchPath) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    // 1 -> 2, 1 -> 3 (fan-out from source)
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 3, 0});
    // Both feed into output
    g.nodes.push_back({4, "blend"});
    g.connections.push_back({2, 0, 4, 0});
    g.connections.push_back({3, 0, 4, 1});
    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    // Select node 2 (one branch) as active
    auto path = ActivePathTracer::trace(r.ir, 2, lib);

    // Path should be: source(1) -> step(2)
    ASSERT_EQ(path.nodes.size(), 2u);
    EXPECT_EQ(path.nodes[0], 1u);
    EXPECT_EQ(path.nodes[1], 2u);
    // Node 3 should NOT be in the path (different branch)
    EXPECT_TRUE(std::find(path.nodes.begin(), path.nodes.end(), 3) == path.nodes.end());
}

TEST(ActivePathTracer, MergePoint) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    g.nodes.push_back({4, "blend"});
    // Diamond: 1->2, 1->3, 2->4, 3->4
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 4, 0});
    g.connections.push_back({3, 0, 4, 1});
    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    // Select node 4 (merge point) as active
    auto path = ActivePathTracer::trace(r.ir, 4, lib);

    // Path should include all 4 nodes: 1, 2, 3, 4
    ASSERT_EQ(path.nodes.size(), 4u);
    EXPECT_EQ(path.nodes[0], 1u);
    // 2 and 3 should be in the path (both feed into 4)
    EXPECT_TRUE(std::find(path.nodes.begin(), path.nodes.end(), 2) != path.nodes.end());
    EXPECT_TRUE(std::find(path.nodes.begin(), path.nodes.end(), 3) != path.nodes.end());
    EXPECT_EQ(path.nodes.back(), 4u);
}

TEST(ActivePathTracer, InvalidNodeReturnsEmpty) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.output_node = 1;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto path = ActivePathTracer::trace(r.ir, 999, lib);
    EXPECT_TRUE(path.nodes.empty());
}

TEST(ActivePathTracer, ZeroNodeReturnsEmpty) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.output_node = 1;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto path = ActivePathTracer::trace(r.ir, 0, lib);
    EXPECT_TRUE(path.nodes.empty());
}
