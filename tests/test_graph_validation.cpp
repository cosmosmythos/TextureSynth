#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
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
