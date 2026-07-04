#include <gtest/gtest.h>
#include <algorithm>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/FusionGroup.hpp"
#include "engine/graphfusion/FusionGroupEmitter.hpp"
#include "engine/graphfusion/FusedGroupCompiler.hpp"

using namespace te;

namespace {

NodeType make_type(const std::string& id, uint32_t n_in, uint32_t n_out,
                   uint32_t pass_count = 1,
                   const std::vector<SocketType>& input_types = {},
                   const std::string& glsl = "",
                   uint32_t n_params = 0) {
    NodeType t;
    t.id = id;
    t.display_name = id;
    t.pass_kind = PassKind::Compute;
    t.pass_count = pass_count;
    t.glsl_function = glsl;
    for (uint32_t i = 0; i < n_in; ++i) {
        Socket s;
        s.name = "in" + std::to_string(i);
        s.type = (i < input_types.size()) ? input_types[i] : SocketType::Vec4;
        t.inputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        Socket s;
        s.name = "out" + std::to_string(i);
        s.type = SocketType::Vec4;
        t.outputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_params; ++i) {
        NodeParam p;
        p.name = "param" + std::to_string(i);
        t.params.push_back(p);
    }
    return t;
}

NodeLibrary make_lib() {
    NodeLibrary lib;
    lib.add_public(make_type("worley",  0, 1, 1, {}, "vec4 node_worley(vec2 uv) { return vec4(0.0); }", 7));
    lib.add_public(make_type("perlin",  0, 1, 1, {}, "vec4 node_perlin(vec2 uv) { return vec4(0.0); }", 6));
    lib.add_public(make_type("blur",    1, 1, 2, {SocketType::Sampler2D}, "vec4 node_blur(vec2 uv, TSTexture tex) { return vec4(0.0); }", 1));
    lib.add_public(make_type("invert",  2, 1, 1, {SocketType::Float, SocketType::Vec4}, "vec4 node_invert(vec2 uv, float mask, vec4 color) { return vec4(0.0); }", 0));
    lib.add_public(make_type("levels",  1, 1, 1, {}, "vec4 node_levels(vec2 uv, vec4 color) { return vec4(0.0); }", 25));
    lib.add_public(make_type("warp",    2, 1, 1, {SocketType::Sampler2D, SocketType::Sampler2D}, "vec4 node_warp(vec2 uv, TSTexture image, TSTexture gradient) { return vec4(0.0); }", 4));
    lib.add_public(make_type("blend",   3, 1, 1, {SocketType::Float, SocketType::Vec4, SocketType::Vec4}, "vec4 node_blend(vec2 uv, float mask, vec4 a, vec4 b) { return vec4(0.0); }", 1));
    return lib;
}

} // namespace

class FusionGroupTest : public ::testing::Test {
protected:
    NodeLibrary lib = make_lib();
    Graph g;
    GraphIR ir;
    fusion::FusionContext ctx;

    std::string node_name(NodeId id) const {
        const auto* n = ir.find(id);
        return n ? n->debug_name : "?";
    }

    void build_graph() {
        g.nodes.push_back({1,  "worley",  ChannelFormat::RGBA, "Worley (Cellular)"});
        g.nodes.push_back({2,  "blur",    ChannelFormat::RGBA, "Blur.001"});
        g.nodes.push_back({3,  "invert",  ChannelFormat::RGBA, "Invert"});
        g.nodes.push_back({8,  "perlin",  ChannelFormat::RGBA, "Perlin Noise.001"});
        g.nodes.push_back({5,  "levels",  ChannelFormat::RGBA, "Levels"});
        g.nodes.push_back({4,  "warp",    ChannelFormat::RGBA, "Warp"});
        g.nodes.push_back({6,  "perlin",  ChannelFormat::RGBA, "Perlin Noise"});
        g.nodes.push_back({10, "worley",  ChannelFormat::RGBA, "Worley (Cellular).001"});
        g.nodes.push_back({11, "invert",  ChannelFormat::RGBA, "Invert.001"});
        g.nodes.push_back({7,  "warp",    ChannelFormat::RGBA, "Warp.001"});
        g.nodes.push_back({12, "invert",  ChannelFormat::RGBA, "Invert.002"});
        g.nodes.push_back({9,  "blend",   ChannelFormat::RGBA, "Blend"});

        g.connections.push_back({1,  0, 2,  0});
        g.connections.push_back({2,  0, 3,  1});
        g.connections.push_back({3,  0, 4,  1});
        g.connections.push_back({5,  0, 4,  0});
        g.connections.push_back({6,  0, 7,  1});
        g.connections.push_back({8,  0, 5,  0});
        g.connections.push_back({4,  0, 9,  2});
        g.connections.push_back({10, 0, 11, 1});
        g.connections.push_back({11, 0, 7,  0});
        g.connections.push_back({7,  0, 12, 1});
        g.connections.push_back({12, 0, 9,  1});

        g.output_node = 9;

        auto r = validate_graph(g, lib);
        ASSERT_TRUE(r.success) << r.error;
        ir = std::move(r.ir);
        ctx = fusion::build_context(ir, lib);
    }
};

TEST_F(FusionGroupTest, GraphBuildsCorrectly) {
    build_graph();
    EXPECT_EQ(ir.eval_order.size(), 12u);
    EXPECT_EQ(ir.output_node, 9u);
}

TEST_F(FusionGroupTest, EvalOrderIsTopological) {
    build_graph();
    auto index_of = [&](NodeId id) -> int {
        for (size_t i = 0; i < ir.eval_order.size(); ++i)
            if (ir.eval_order[i] == id) return (int)i;
        return -1;
    };
    for (const auto& c : ir.connections)
        EXPECT_LT(index_of(c.src_node), index_of(c.dst_node));
}

TEST_F(FusionGroupTest, EmptyGraphReturnsEmpty) {
    GraphIR empty_ir;
    fusion::FusionContext empty_ctx;
    auto result = fusion::group_nodes(empty_ir, empty_ctx);
    EXPECT_TRUE(result.groups.empty());
}

TEST_F(FusionGroupTest, ExpandMultipassDoublesBlur) {
    build_graph();
    auto expanded = fusion::expand_multipass(ir.eval_order, ctx);
    EXPECT_EQ(expanded.size(), 13u);
    EXPECT_EQ(expanded[1].node_id, 2u);
    EXPECT_EQ(expanded[1].pass_index, 0u);
    EXPECT_EQ(expanded[2].node_id, 2u);
    EXPECT_EQ(expanded[2].pass_index, 1u);
}

TEST_F(FusionGroupTest, GroupNodesRuns) {
    build_graph();

    // GraphIR dump
    std::cout << "\n=== GraphIR: nodes ===\n";
    for (const auto& vn : ir.nodes) {
        std::cout << "  id=" << vn.id << " type=" << vn.type_id
                  << " name=" << vn.debug_name << "\n";
    }
    std::cout << "\n=== GraphIR: connections ===\n";
    for (const auto& c : ir.connections) {
        auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type[c.dst_node] : nullptr;
        std::string socket_type = "?";
        if (st && c.dst_socket < st->inputs.size())
            socket_type = st->inputs[c.dst_socket].type == SocketType::Vec4 ? "Vec4" :
                          st->inputs[c.dst_socket].type == SocketType::Sampler2D ? "Sampler2D" :
                          st->inputs[c.dst_socket].type == SocketType::Float ? "Float" : "?";
        std::cout << "  " << node_name(c.src_node) << "(" << c.src_node << ") socket" << c.src_socket
                  << " -> " << node_name(c.dst_node) << "(" << c.dst_node << ") socket" << c.dst_socket
                  << " [" << socket_type << "]\n";
    }
    std::cout << "\n=== GraphIR: eval_order ===\n";
    for (size_t i = 0; i < ir.eval_order.size(); ++i) {
        NodeId id = ir.eval_order[i];
        const auto* type = ctx.node_type.count(id) ? ctx.node_type[id] : nullptr;
        std::string inputs_str;
        if (type) {
            for (size_t j = 0; j < type->inputs.size(); ++j) {
                if (j) inputs_str += ", ";
                inputs_str += type->inputs[j].type == SocketType::Vec4 ? "Vec4" :
                              type->inputs[j].type == SocketType::Sampler2D ? "Sampler2D" :
                              type->inputs[j].type == SocketType::Float ? "Float" : "?";
            }
        }
        std::cout << "  [" << i << "] " << node_name(id) << "(" << id << ")"
                  << " inputs=[" << inputs_str << "]"
                  << " params=" << (type ? type->params.size() : 0) << "\n";
    }

    // Expanded order
    auto expanded = fusion::expand_multipass(ir.eval_order, ctx);
    std::cout << "\n=== Expanded order ===\n";
    for (size_t i = 0; i < expanded.size(); ++i) {
        auto& e = expanded[i];
        std::cout << "  [" << i << "] " << node_name(e.node_id) << " (id=" << e.node_id << ")"
                  << " pass=" << e.pass_index << "/" << e.pass_count << "\n";
    }

    // Grouping decisions
    std::cout << "\n=== Grouping decisions ===\n";
    for (size_t i = 0; i + 1 < expanded.size(); ++i) {
        NodeId n1 = expanded[i].node_id;
        NodeId n2 = expanded[i + 1].node_id;
        auto ct = fusion::get_connection_type(n1, n2, ctx);
        std::string type_str = "none";
        if (ct == SocketType::Vec4) type_str = "Vec4";
        else if (ct == SocketType::Sampler2D) type_str = "Sampler2D";
        else if (ct == SocketType::Float) type_str = "Float";
        bool conn = fusion::is_connected(n1, n2, ctx);
        std::cout << "  " << node_name(n1) << "(" << n1 << ") -> "
                  << node_name(n2) << "(" << n2 << "): "
                  << type_str << (conn ? " [MERGE]" : " [BREAK]") << "\n";
    }

    auto result = fusion::group_nodes(ir, ctx);
    std::cout << "\n=== Before merge (" << result.groups.size() << " groups) ===\n";
    for (size_t g = 0; g < result.groups.size(); ++g) {
        std::cout << "  Group " << g << ": [";
        for (size_t n = 0; n < result.groups[g].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(result.groups[g].nodes[n]) << "(" << result.groups[g].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    fusion::split_at_sampler2d_sources(result, ctx);
    fusion::merge_groups(result, ctx);

    std::cout << "\n=== After merge (" << result.groups.size() << " groups) ===\n";
    for (size_t g = 0; g < result.groups.size(); ++g) {
        std::cout << "  Group " << g << ": [";
        for (size_t n = 0; n < result.groups[g].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(result.groups[g].nodes[n]) << "(" << result.groups[g].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    // Show cross-group connections
    std::cout << "\n=== Cross-group connections ===\n";
    for (const auto& c : ir.connections) {
        int src_group = -1, dst_group = -1;
        for (size_t g = 0; g < result.groups.size(); ++g) {
            for (NodeId n : result.groups[g].nodes) {
                if (n == c.src_node) src_group = (int)g;
                if (n == c.dst_node) dst_group = (int)g;
            }
        }
        if (src_group != dst_group) {
            auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type[c.dst_node] : nullptr;
            std::string socket_type = "?";
            if (st && c.dst_socket < st->inputs.size())
                socket_type = st->inputs[c.dst_socket].type == SocketType::Vec4 ? "Vec4" :
                              st->inputs[c.dst_socket].type == SocketType::Sampler2D ? "Sampler2D" : "?";
            std::cout << "  " << node_name(c.src_node) << "(" << c.src_node << ") g" << src_group
                      << " -> " << node_name(c.dst_node) << "(" << c.dst_node << ") g" << dst_group
                      << " [" << socket_type << "]\n";
        }
    }

    fusion::compute_external_inputs(result, ctx);

    std::cout << "\n=== External inputs ===\n";
    for (size_t g = 0; g < result.groups.size(); ++g) {
        if (result.groups[g].external_inputs.empty()) continue;
        std::cout << "  Group " << g << ":\n";
        for (const auto& ext : result.groups[g].external_inputs) {
            std::cout << "    slot=" << ext.slot << " " << node_name(ext.src_node)
                      << "(" << ext.src_node << ") socket" << ext.src_socket
                      << " -> " << node_name(ext.dst_node)
                      << "(" << ext.dst_node << ") socket" << ext.dst_socket << "\n";
        }
    }

    EXPECT_EQ(result.groups.size(), 7u);
    EXPECT_EQ(result.groups[0].nodes.size(), 1u);
    EXPECT_EQ(result.groups[0].nodes[0], 1u);
    EXPECT_EQ(result.groups[1].nodes.size(), 1u);
    EXPECT_EQ(result.groups[1].nodes[0], 2u);
    EXPECT_EQ(result.groups[2].nodes.size(), 2u);
    EXPECT_EQ(result.groups[2].nodes[0], 2u);
    EXPECT_EQ(result.groups[2].nodes[1], 3u);
    EXPECT_EQ(result.groups[3].nodes.size(), 2u);
    EXPECT_EQ(result.groups[3].nodes[0], 8u);
    EXPECT_EQ(result.groups[3].nodes[1], 5u);
    EXPECT_EQ(result.groups[4].nodes.size(), 1u);
    EXPECT_EQ(result.groups[4].nodes[0], 6u);
    EXPECT_EQ(result.groups[5].nodes.size(), 2u);
    EXPECT_EQ(result.groups[5].nodes[0], 10u);
    EXPECT_EQ(result.groups[5].nodes[1], 11u);
    EXPECT_EQ(result.groups[6].nodes.size(), 4u);
    EXPECT_EQ(result.groups[6].nodes[0], 4u);
    EXPECT_EQ(result.groups[6].nodes[1], 7u);
    EXPECT_EQ(result.groups[6].nodes[2], 12u);
    EXPECT_EQ(result.groups[6].nodes[3], 9u);
}

TEST_F(FusionGroupTest, EmitGroupsProducesGLSL) {
    build_graph();
    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    for (size_t g = 0; g < fused.groups.size(); ++g) {
        const auto& group = fused.groups[g];

        std::cout << "\n=== Group " << g << " ===\n";
        std::cout << "  Nodes:";
        for (NodeId n : group.nodes) std::cout << " " << node_name(n) << "(" << n << ")";
        std::cout << "\n";
        if (!group.external_inputs.empty()) {
            std::cout << "  External inputs:\n";
            for (const auto& ext : group.external_inputs) {
                std::cout << "    slot " << ext.slot << ": "
                          << node_name(ext.src_node) << "(" << ext.src_node << ") output" << ext.src_socket
                          << " -> " << node_name(ext.dst_node) << "(" << ext.dst_node << ") input" << ext.dst_socket
                          << "\n";
            }
        }

        auto emit = fusion::emit_group(group, ir, ctx, g);
        EXPECT_TRUE(emit.ok()) << "Group " << g << ": " << emit.error;

        if (g == fused.groups.size() - 1)
            std::cout << "\n  FULL SHADER:\n" << emit.source << "\n";

        auto main_pos = emit.source.find("void main()");
        if (main_pos != std::string::npos)
            std::cout << "  GLSL main:\n" << emit.source.substr(main_pos) << "\n";
    }
}

// ============================================================================
// FusedGroupCompiler tests
// ============================================================================

TEST_F(FusionGroupTest, CompileGroupsProducesGLSL) {
    build_graph();
    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;
    EXPECT_EQ(compiled.groups.size(), fused.groups.size());

    for (size_t i = 0; i < compiled.groups.size(); ++i) {
        EXPECT_TRUE(compiled.groups[i].ok()) << "group " << i << ": " << compiled.groups[i].error;
        EXPECT_FALSE(compiled.groups[i].glsl.empty()) << "group " << i << " has no GLSL";
        EXPECT_FALSE(compiled.groups[i].glsl.find("void main()") == std::string::npos)
            << "group " << i << " missing main()";
    }
}

TEST_F(FusionGroupTest, CompileGroupsParamLayout) {
    build_graph();
    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    EXPECT_EQ(compiled.groups[0].param_floats, 7u);
    EXPECT_EQ(compiled.groups[1].param_floats, 1u);
    EXPECT_EQ(compiled.groups[2].param_floats, 2u);
    EXPECT_EQ(compiled.groups[3].param_floats, 31u);
    EXPECT_EQ(compiled.groups[6].param_floats, 11u);
}

TEST_F(FusionGroupTest, CompileGroupsExternalInputs) {
    build_graph();
    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    EXPECT_TRUE(compiled.groups[0].external_inputs.empty());
    EXPECT_EQ(compiled.groups[1].external_inputs.size(), 1u);
    EXPECT_EQ(compiled.groups[1].external_inputs[0].src_node, 1u);
    EXPECT_EQ(compiled.groups[6].external_inputs.size(), 4u);
}

TEST_F(FusionGroupTest, CompileGroupsOutputNode) {
    build_graph();
    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    auto& last = compiled.groups.back();
    EXPECT_EQ(last.output_node, 9u);
}

TEST_F(FusionGroupTest, CompileGroupsParamBaseSlot) {
    build_graph();
    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    for (size_t i = 0; i < compiled.groups.size(); ++i) {
        EXPECT_NE(compiled.groups[i].param_base_slot, UINT32_MAX)
            << "group " << i << " has invalid param_base_slot";
    }
}

TEST_F(FusionGroupTest, CompileGroupsEmptyGraphReturnsEmpty) {
    GraphIR empty_ir;
    fusion::FusionContext empty_ctx;
    fusion::FusionGroupBundle empty_bundle;
    auto compiled = fusion::compile_groups(empty_bundle, empty_ir, empty_ctx);
    EXPECT_TRUE(compiled.ok());
    EXPECT_TRUE(compiled.groups.empty());
}

// ============================================================================
// Diamond pattern: A feeds B and C, B+C feed D
//
//   Perlin(A) --+--> Invert(B) --+
//                |                +--> Blend(D)
//                +--> Invert(C) --+
//
// All connections are Vec4. Should merge into one group.
// ============================================================================

TEST_F(FusionGroupTest, DiamondPattern) {
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "Perlin"});
    g.nodes.push_back({2, "invert", ChannelFormat::RGBA, "Invert"});
    g.nodes.push_back({3, "invert", ChannelFormat::RGBA, "Invert.001"});
    g.nodes.push_back({4, "blend",  ChannelFormat::RGBA, "Blend"});

    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({1, 0, 3, 1});
    g.connections.push_back({2, 0, 4, 1});
    g.connections.push_back({3, 0, 4, 2});

    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    ir = std::move(r.ir);
    ctx = fusion::build_context(ir, lib);

    // GraphIR dump
    std::cout << "\n=== Diamond GraphIR: nodes ===\n";
    for (const auto& vn : ir.nodes) {
        std::cout << "  id=" << vn.id << " type=" << vn.type_id
                  << " name=" << vn.debug_name << "\n";
    }
    std::cout << "\n=== Diamond GraphIR: connections ===\n";
    for (const auto& c : ir.connections) {
        auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type[c.dst_node] : nullptr;
        std::string socket_type = "?";
        if (st && c.dst_socket < st->inputs.size())
            socket_type = st->inputs[c.dst_socket].type == SocketType::Vec4 ? "Vec4" :
                          st->inputs[c.dst_socket].type == SocketType::Sampler2D ? "Sampler2D" :
                          st->inputs[c.dst_socket].type == SocketType::Float ? "Float" : "?";
        std::cout << "  " << node_name(c.src_node) << "(" << c.src_node << ") socket" << c.src_socket
                  << " -> " << node_name(c.dst_node) << "(" << c.dst_node << ") socket" << c.dst_socket
                  << " [" << socket_type << "]\n";
    }
    std::cout << "\n=== Diamond GraphIR: eval_order ===\n";
    for (size_t i = 0; i < ir.eval_order.size(); ++i) {
        NodeId id = ir.eval_order[i];
        const auto* type = ctx.node_type.count(id) ? ctx.node_type[id] : nullptr;
        std::string inputs_str;
        if (type) {
            for (size_t j = 0; j < type->inputs.size(); ++j) {
                if (j) inputs_str += ", ";
                inputs_str += type->inputs[j].type == SocketType::Vec4 ? "Vec4" :
                              type->inputs[j].type == SocketType::Sampler2D ? "Sampler2D" :
                              type->inputs[j].type == SocketType::Float ? "Float" : "?";
            }
        }
        std::cout << "  [" << i << "] " << node_name(id) << "(" << id << ")"
                  << " inputs=[" << inputs_str << "]"
                  << " params=" << (type ? type->params.size() : 0) << "\n";
    }

    // Grouping decisions
    auto expanded = fusion::expand_multipass(ir.eval_order, ctx);
    std::cout << "\n=== Diamond grouping decisions ===\n";
    for (size_t i = 0; i + 1 < expanded.size(); ++i) {
        NodeId n1 = expanded[i].node_id;
        NodeId n2 = expanded[i + 1].node_id;
        auto ct = fusion::get_connection_type(n1, n2, ctx);
        std::string type_str = "none";
        if (ct == SocketType::Vec4) type_str = "Vec4";
        else if (ct == SocketType::Sampler2D) type_str = "Sampler2D";
        else if (ct == SocketType::Float) type_str = "Float";
        bool conn = fusion::is_connected(n1, n2, ctx);
        std::cout << "  " << node_name(n1) << "(" << n1 << ") -> "
                  << node_name(n2) << "(" << n2 << "): "
                  << type_str << (conn ? " [MERGE]" : " [BREAK]") << "\n";
    }

    auto fused = fusion::group_nodes(ir, ctx);
    std::cout << "\n=== Diamond before merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);

    std::cout << "\n=== Diamond after merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    // Cross-group connections
    std::cout << "\n=== Diamond cross-group connections ===\n";
    for (const auto& c : ir.connections) {
        int src_group = -1, dst_group = -1;
        for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
            for (NodeId n : fused.groups[gi].nodes) {
                if (n == c.src_node) src_group = (int)gi;
                if (n == c.dst_node) dst_group = (int)gi;
            }
        }
        if (src_group != dst_group) {
            auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type[c.dst_node] : nullptr;
            std::string socket_type = "?";
            if (st && c.dst_socket < st->inputs.size())
                socket_type = st->inputs[c.dst_socket].type == SocketType::Vec4 ? "Vec4" :
                              st->inputs[c.dst_socket].type == SocketType::Sampler2D ? "Sampler2D" : "?";
            std::cout << "  " << node_name(c.src_node) << "(" << c.src_node << ") g" << src_group
                      << " -> " << node_name(c.dst_node) << "(" << c.dst_node << ") g" << dst_group
                      << " [" << socket_type << "]\n";
        }
    }

    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n=== Diamond external inputs ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        if (fused.groups[gi].external_inputs.empty()) continue;
        std::cout << "  Group " << gi << ":\n";
        for (const auto& ext : fused.groups[gi].external_inputs) {
            std::cout << "    slot=" << ext.slot << " " << node_name(ext.src_node)
                      << "(" << ext.src_node << ") socket" << ext.src_socket
                      << " -> " << node_name(ext.dst_node)
                      << "(" << ext.dst_node << ") socket" << ext.dst_socket << "\n";
        }
    }

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    for (size_t i = 0; i < compiled.groups.size(); ++i) {
        EXPECT_FALSE(compiled.groups[i].glsl.empty()) << "group " << i << " has no GLSL";
        EXPECT_NE(compiled.groups[i].glsl.find("void main()"), std::string::npos)
            << "group " << i << " missing main()";
    }

    std::vector<NodeId> all_nodes;
    for (auto& cg : compiled.groups)
        all_nodes.insert(all_nodes.end(), cg.nodes.begin(), cg.nodes.end());
    std::sort(all_nodes.begin(), all_nodes.end());
    EXPECT_EQ(all_nodes.size(), 4u);
    EXPECT_EQ(compiled.groups.back().output_node, 4u);
}

// ============================================================================
// Fan-out: A feeds B and C, both feed D (converging fan-out)
//
//   Perlin(A) --+--> Invert(B) --+
//                |                +--> Blend(D)
//                +--> Invert(C) --+
//
// Same as DiamondPattern — the only valid fan-out where all nodes
// are predecessors of the output.
// ============================================================================

TEST_F(FusionGroupTest, FanOutConverges) {
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "Perlin"});
    g.nodes.push_back({2, "invert", ChannelFormat::RGBA, "Invert"});
    g.nodes.push_back({3, "invert", ChannelFormat::RGBA, "Invert.001"});
    g.nodes.push_back({4, "blend",  ChannelFormat::RGBA, "Blend"});

    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({1, 0, 3, 1});
    g.connections.push_back({2, 0, 4, 1});
    g.connections.push_back({3, 0, 4, 2});

    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    ir = std::move(r.ir);
    ctx = fusion::build_context(ir, lib);

    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n=== FanOutConverges: eval_order ===\n";
    for (size_t i = 0; i < ir.eval_order.size(); ++i) {
        NodeId id = ir.eval_order[i];
        const auto* type = ctx.node_type.count(id) ? ctx.node_type[id] : nullptr;
        std::string inputs_str;
        if (type) {
            for (size_t j = 0; j < type->inputs.size(); ++j) {
                if (j) inputs_str += ", ";
                inputs_str += type->inputs[j].type == SocketType::Vec4 ? "Vec4" :
                              type->inputs[j].type == SocketType::Sampler2D ? "Sampler2D" :
                              type->inputs[j].type == SocketType::Float ? "Float" : "?";
            }
        }
        std::cout << "  [" << i << "] " << node_name(id) << "(" << id << ")"
                  << " inputs=[" << inputs_str << "]\n";
    }

    auto expanded = fusion::expand_multipass(ir.eval_order, ctx);
    std::cout << "\n=== FanOutConverges grouping decisions ===\n";
    for (size_t i = 0; i + 1 < expanded.size(); ++i) {
        NodeId n1 = expanded[i].node_id;
        NodeId n2 = expanded[i + 1].node_id;
        auto ct = fusion::get_connection_type(n1, n2, ctx);
        std::string type_str = "none";
        if (ct == SocketType::Vec4) type_str = "Vec4";
        else if (ct == SocketType::Sampler2D) type_str = "Sampler2D";
        bool conn = fusion::is_connected(n1, n2, ctx);
        std::cout << "  " << node_name(n1) << "(" << n1 << ") -> "
                  << node_name(n2) << "(" << n2 << "): "
                  << type_str << (conn ? " [MERGE]" : " [BREAK]") << "\n";
    }

    std::cout << "\n=== FanOutConverges before merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);

    std::cout << "\n=== FanOutConverges after merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    fusion::compute_external_inputs(fused, ctx);

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    // All 4 nodes in one group (all Vec4 connected, no Sampler2D boundaries)
    EXPECT_EQ(fused.groups.size(), 1u);
    EXPECT_EQ(compiled.groups.back().output_node, 4u);
}

// ============================================================================
// Fan-out with Sampler2D: A feeds B (Vec4) and C (Sampler2D), both feed D
//
//   Perlin(A) --+--> Invert(B) --+
//                |  [Vec4]        +--> Blend(D)
//                +--> Warp(C) ----+  [Sampler2D]
//
// A→B is Vec4, A→C is Sampler2D. B and C can't be in the same group as A
// if Sampler2D boundary exists.
// ============================================================================

TEST_F(FusionGroupTest, FanOutWithSampler2D) {
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "Perlin"});
    g.nodes.push_back({2, "invert", ChannelFormat::RGBA, "Invert"});
    g.nodes.push_back({3, "warp",   ChannelFormat::RGBA, "Warp"});
    g.nodes.push_back({4, "blend",  ChannelFormat::RGBA, "Blend"});

    g.connections.push_back({1, 0, 2, 1});  // A → B Vec4
    g.connections.push_back({1, 0, 3, 0});  // A → C Sampler2D (image)
    g.connections.push_back({2, 0, 4, 1});  // B → D Vec4 (a)
    g.connections.push_back({3, 0, 4, 2});  // C → D Vec4 (b)

    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    ir = std::move(r.ir);
    ctx = fusion::build_context(ir, lib);

    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n=== FanOutWithSampler2D: eval_order ===\n";
    for (size_t i = 0; i < ir.eval_order.size(); ++i) {
        NodeId id = ir.eval_order[i];
        const auto* type = ctx.node_type.count(id) ? ctx.node_type[id] : nullptr;
        std::string inputs_str;
        if (type) {
            for (size_t j = 0; j < type->inputs.size(); ++j) {
                if (j) inputs_str += ", ";
                inputs_str += type->inputs[j].type == SocketType::Vec4 ? "Vec4" :
                              type->inputs[j].type == SocketType::Sampler2D ? "Sampler2D" :
                              type->inputs[j].type == SocketType::Float ? "Float" : "?";
            }
        }
        std::cout << "  [" << i << "] " << node_name(id) << "(" << id << ")"
                  << " inputs=[" << inputs_str << "]\n";
    }

    auto expanded = fusion::expand_multipass(ir.eval_order, ctx);
    std::cout << "\n=== FanOutWithSampler2D grouping decisions ===\n";
    for (size_t i = 0; i + 1 < expanded.size(); ++i) {
        NodeId n1 = expanded[i].node_id;
        NodeId n2 = expanded[i + 1].node_id;
        auto ct = fusion::get_connection_type(n1, n2, ctx);
        std::string type_str = "none";
        if (ct == SocketType::Vec4) type_str = "Vec4";
        else if (ct == SocketType::Sampler2D) type_str = "Sampler2D";
        bool conn = fusion::is_connected(n1, n2, ctx);
        std::cout << "  " << node_name(n1) << "(" << n1 << ") -> "
                  << node_name(n2) << "(" << n2 << "): "
                  << type_str << (conn ? " [MERGE]" : " [BREAK]") << "\n";
    }

    std::cout << "\n=== FanOutWithSampler2D before merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    std::cout << "\n=== FanOutWithSampler2D after merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n=== FanOutWithSampler2D external inputs ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        if (fused.groups[gi].external_inputs.empty()) continue;
        std::cout << "  Group " << gi << ":\n";
        for (const auto& ext : fused.groups[gi].external_inputs) {
            std::cout << "    slot=" << ext.slot << " " << node_name(ext.src_node)
                      << "(" << ext.src_node << ") socket" << ext.src_socket
                      << " -> " << node_name(ext.dst_node)
                      << "(" << ext.dst_node << ") socket" << ext.dst_socket << "\n";
        }
    }

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    // A+B in one group (Vec4), C separate (Sampler2D from A), D merges with B
    // Perlin(1) has Sampler2D consumer Warp(3) → split separates Perlin.
    // Invert(2) re-merges with Warp+Blend via Invert→Blend Vec4.
    // Expected: group0=[Perlin], group1=[Warp,Invert,Blend]
    EXPECT_EQ(fused.groups.size(), 2u);
    EXPECT_EQ(compiled.groups.back().output_node, 4u);

    // Perlin(1) should be external input to Warp's group
    bool found_ext = false;
    for (auto& fg : fused.groups)
        for (auto& ext : fg.external_inputs)
            if (ext.src_node == 1 && ext.dst_node == 3)
                found_ext = true;
    EXPECT_TRUE(found_ext) << "Perlin(1) should be external input to Warp(3) via Sampler2D";
}

// ============================================================================
// Diamond with chain in one branch:
//
//   Perlin(A) --+--> Invert(B) --> Invert.001(C) --+
//                |                                   +--> Blend(D)
//                +--> Invert.002(E) ----------------+
//
// A feeds B (Vec4) and E (Vec4). B feeds C (Vec4). C and E feed D.
// ============================================================================

TEST_F(FusionGroupTest, DiamondWithChain) {
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "Perlin"});
    g.nodes.push_back({2, "invert", ChannelFormat::RGBA, "Invert"});
    g.nodes.push_back({3, "invert", ChannelFormat::RGBA, "Invert.001"});
    g.nodes.push_back({5, "invert", ChannelFormat::RGBA, "Invert.002"});
    g.nodes.push_back({4, "blend",  ChannelFormat::RGBA, "Blend"});

    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({1, 0, 5, 1});
    g.connections.push_back({2, 0, 3, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({5, 0, 4, 2});

    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    ir = std::move(r.ir);
    ctx = fusion::build_context(ir, lib);

    // GraphIR dump
    std::cout << "\n=== DiamondChain GraphIR: nodes ===\n";
    for (const auto& vn : ir.nodes) {
        std::cout << "  id=" << vn.id << " type=" << vn.type_id
                  << " name=" << vn.debug_name << "\n";
    }
    std::cout << "\n=== DiamondChain GraphIR: connections ===\n";
    for (const auto& c : ir.connections) {
        auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type[c.dst_node] : nullptr;
        std::string socket_type = "?";
        if (st && c.dst_socket < st->inputs.size())
            socket_type = st->inputs[c.dst_socket].type == SocketType::Vec4 ? "Vec4" :
                          st->inputs[c.dst_socket].type == SocketType::Sampler2D ? "Sampler2D" :
                          st->inputs[c.dst_socket].type == SocketType::Float ? "Float" : "?";
        std::cout << "  " << node_name(c.src_node) << "(" << c.src_node << ") socket" << c.src_socket
                  << " -> " << node_name(c.dst_node) << "(" << c.dst_node << ") socket" << c.dst_socket
                  << " [" << socket_type << "]\n";
    }
    std::cout << "\n=== DiamondChain GraphIR: eval_order ===\n";
    for (size_t i = 0; i < ir.eval_order.size(); ++i) {
        NodeId id = ir.eval_order[i];
        const auto* type = ctx.node_type.count(id) ? ctx.node_type[id] : nullptr;
        std::string inputs_str;
        if (type) {
            for (size_t j = 0; j < type->inputs.size(); ++j) {
                if (j) inputs_str += ", ";
                inputs_str += type->inputs[j].type == SocketType::Vec4 ? "Vec4" :
                              type->inputs[j].type == SocketType::Sampler2D ? "Sampler2D" :
                              type->inputs[j].type == SocketType::Float ? "Float" : "?";
            }
        }
        std::cout << "  [" << i << "] " << node_name(id) << "(" << id << ")"
                  << " inputs=[" << inputs_str << "]"
                  << " params=" << (type ? type->params.size() : 0) << "\n";
    }

    // Grouping decisions
    auto expanded = fusion::expand_multipass(ir.eval_order, ctx);
    std::cout << "\n=== DiamondChain grouping decisions ===\n";
    for (size_t i = 0; i + 1 < expanded.size(); ++i) {
        NodeId n1 = expanded[i].node_id;
        NodeId n2 = expanded[i + 1].node_id;
        auto ct = fusion::get_connection_type(n1, n2, ctx);
        std::string type_str = "none";
        if (ct == SocketType::Vec4) type_str = "Vec4";
        else if (ct == SocketType::Sampler2D) type_str = "Sampler2D";
        else if (ct == SocketType::Float) type_str = "Float";
        bool conn = fusion::is_connected(n1, n2, ctx);
        std::cout << "  " << node_name(n1) << "(" << n1 << ") -> "
                  << node_name(n2) << "(" << n2 << "): "
                  << type_str << (conn ? " [MERGE]" : " [BREAK]") << "\n";
    }

    auto fused = fusion::group_nodes(ir, ctx);
    std::cout << "\n=== DiamondChain before merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);

    std::cout << "\n=== DiamondChain after merge (" << fused.groups.size() << " groups) ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n]) << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    // Cross-group connections
    std::cout << "\n=== DiamondChain cross-group connections ===\n";
    for (const auto& c : ir.connections) {
        int src_group = -1, dst_group = -1;
        for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
            for (NodeId n : fused.groups[gi].nodes) {
                if (n == c.src_node) src_group = (int)gi;
                if (n == c.dst_node) dst_group = (int)gi;
            }
        }
        if (src_group != dst_group) {
            auto st = ctx.node_type.count(c.dst_node) ? ctx.node_type[c.dst_node] : nullptr;
            std::string socket_type = "?";
            if (st && c.dst_socket < st->inputs.size())
                socket_type = st->inputs[c.dst_socket].type == SocketType::Vec4 ? "Vec4" :
                              st->inputs[c.dst_socket].type == SocketType::Sampler2D ? "Sampler2D" : "?";
            std::cout << "  " << node_name(c.src_node) << "(" << c.src_node << ") g" << src_group
                      << " -> " << node_name(c.dst_node) << "(" << c.dst_node << ") g" << dst_group
                      << " [" << socket_type << "]\n";
        }
    }

    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n=== DiamondChain external inputs ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        if (fused.groups[gi].external_inputs.empty()) continue;
        std::cout << "  Group " << gi << ":\n";
        for (const auto& ext : fused.groups[gi].external_inputs) {
            std::cout << "    slot=" << ext.slot << " " << node_name(ext.src_node)
                      << "(" << ext.src_node << ") socket" << ext.src_socket
                      << " -> " << node_name(ext.dst_node)
                      << "(" << ext.dst_node << ") socket" << ext.dst_socket << "\n";
        }
    }

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    std::vector<NodeId> all_nodes;
    for (auto& cg : compiled.groups)
        all_nodes.insert(all_nodes.end(), cg.nodes.begin(), cg.nodes.end());
    std::sort(all_nodes.begin(), all_nodes.end());
    EXPECT_EQ(all_nodes.size(), 5u);
    EXPECT_EQ(compiled.groups.back().output_node, 4u);

    bool found_chain = false;
    for (auto& cg : compiled.groups) {
        bool has_b = false, has_c = false;
        for (NodeId n : cg.nodes) {
            if (n == 2) has_b = true;
            if (n == 3) has_c = true;
        }
        if (has_b && has_c) found_chain = true;
    }
    EXPECT_TRUE(found_chain) << "B(2) and C(3) should be in the same group (Vec4 chain)";
}

// ============================================================================
// Sampler2D diamond: C consumes A via Sampler2D (warp), not Vec4
//
//   Perlin(A) --+--> Invert(B) --+--> Blend(D)
//                |                |
//                +--> Warp(C) ----+  (C takes A as Sampler2D image)
//
// warp inputs: [Sampler2D image, Sampler2D gradient]
// C takes A as Sampler2D, so is_connected(A,C) = false.
// A groups with B. C is separate. D merges with B.
// ============================================================================

TEST_F(FusionGroupTest, Sampler2DDiamond) {
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "Perlin"});
    g.nodes.push_back({2, "invert", ChannelFormat::RGBA, "Invert"});
    g.nodes.push_back({3, "warp",   ChannelFormat::RGBA, "Warp"});
    g.nodes.push_back({4, "blend",  ChannelFormat::RGBA, "Blend"});

    // A(1) -> B(2) socket 1 (Vec4 color) — Vec4 connected
    // A(1) -> C(3) socket 0 (Sampler2D image) — Sampler2D, NOT Vec4 connected
    // B(2) -> D(4) socket 1 (Vec4 a) — Vec4 connected
    // C(3) -> D(4) socket 2 (Vec4 b) — Vec4 connected
    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 4, 1});
    g.connections.push_back({3, 0, 4, 2});

    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    ir = std::move(r.ir);
    ctx = fusion::build_context(ir, lib);

    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n=== Sampler2DDiamond: groups ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n])
                      << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
        for (const auto& ext : fused.groups[gi].external_inputs) {
            std::cout << "    ext: " << node_name(ext.src_node)
                      << "(" << ext.src_node << ") -> "
                      << node_name(ext.dst_node)
                      << "(" << ext.dst_node << ") slot=" << ext.slot << "\n";
        }
    }

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    // Perlin(1) has Sampler2D consumer Warp(3) → split separates Perlin.
    // Invert(2) re-merges with Warp+Blend via Invert→Blend Vec4.
    // Expected: group0=[Perlin], group1=[Warp,Invert,Blend]
    EXPECT_EQ(fused.groups.size(), 2u);

    // A(1) should appear as external input in C's group (Sampler2D)
    bool found_ext = false;
    for (auto& fg : fused.groups)
        for (auto& ext : fg.external_inputs)
            if (ext.src_node == 1 && ext.dst_node == 3)
                found_ext = true;
    EXPECT_TRUE(found_ext)
        << "Perlin(1) should be external input to Warp(3) via Sampler2D";

    EXPECT_EQ(compiled.groups.back().output_node, 4u);
}

// ============================================================================
// Sampler2D-only diamond: BOTH branches use Sampler2D from A
//
//   Perlin(A) --+--> Warp(B) --+--> Blend(D)
//                |              |
//                +--> Warp(C) --+  (both take A as Sampler2D image)
//
// Neither B nor C are Vec4-connected to A, so they can't merge with A.
// B and C each get A as external Sampler2D input.
// ============================================================================

TEST_F(FusionGroupTest, Sampler2DOnlyDiamond) {
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "Perlin"});
    g.nodes.push_back({2, "warp",   ChannelFormat::RGBA, "Warp"});
    g.nodes.push_back({3, "warp",   ChannelFormat::RGBA, "Warp.001"});
    g.nodes.push_back({4, "blend",  ChannelFormat::RGBA, "Blend"});

    // A(1) -> B(3) socket 0 (Sampler2D image)
    // A(1) -> C(3) socket 0 (Sampler2D image)
    // B(2) -> D(4) socket 1 (Vec4 a)
    // C(3) -> D(4) socket 2 (Vec4 b)
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 4, 1});
    g.connections.push_back({3, 0, 4, 2});

    g.output_node = 4;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    ir = std::move(r.ir);
    ctx = fusion::build_context(ir, lib);

    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);
    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n=== Sampler2DOnlyDiamond: groups ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n])
                      << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
        for (const auto& ext : fused.groups[gi].external_inputs) {
            std::cout << "    ext: " << node_name(ext.src_node)
                      << "(" << ext.src_node << ") -> "
                      << node_name(ext.dst_node)
                      << "(" << ext.dst_node << ") slot=" << ext.slot << "\n";
        }
    }

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    // No Vec4 connections from A, so A can't merge with B or C.
    // B and C are separate groups, each with A as external Sampler2D.
    // D merges with whichever comes first (B or C).

    // A(1) should be external input in BOTH B's and C's groups
    int ext_a_count = 0;
    for (auto& cg : compiled.groups)
        for (auto& ext : cg.external_inputs)
            if (ext.src_node == 1) ++ext_a_count;
    EXPECT_EQ(ext_a_count, 2)
        << "Perlin(1) should be external input to both Warp groups";

    EXPECT_EQ(compiled.groups.back().output_node, 4u);
}

// ============================================================================
// Levels diamond: Levels feeds both Warp (Sampler2D) and Invert (Vec4)
//
//   Perlin(1) --> Levels(2) --[S2D]--> Warp(3) --[Vec4]--> Blend(5)
//                     |
//                     +-----[Vec4]----> Invert(4) --[Vec4]--> Blend(5)
//
// Levels→Warp is Sampler2D → breaks into separate group.
// Levels→Invert is Vec4 → merge with Levels.
// Warp→Blend is Vec4 cross-group → MUST be tracked as external input.
// ============================================================================

TEST_F(FusionGroupTest, LevelsFanOut) {
    g.nodes.push_back({1, "perlin",  ChannelFormat::RGBA, "Perlin"});
    g.nodes.push_back({2, "levels",  ChannelFormat::RGBA, "Levels"});
    g.nodes.push_back({3, "warp",    ChannelFormat::RGBA, "Warp"});
    g.nodes.push_back({4, "invert",  ChannelFormat::RGBA, "Invert"});
    g.nodes.push_back({5, "blend",   ChannelFormat::RGBA, "Blend"});

    // Perlin(1) → Levels(2) socket 0 (Vec4)
    g.connections.push_back({1, 0, 2, 0});
    // Levels(2) → Warp(3) socket 0 (Sampler2D image)
    g.connections.push_back({2, 0, 3, 0});
    // Levels(2) → Invert(4) socket 1 (Vec4 color)
    g.connections.push_back({2, 0, 4, 1});
    // Warp(3) → Blend(5) socket 2 (Vec4 b)
    g.connections.push_back({3, 0, 5, 2});
    // Invert(4) → Blend(5) socket 1 (Vec4 a)
    g.connections.push_back({4, 0, 5, 1});

    g.output_node = 5;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    ir = std::move(r.ir);
    ctx = fusion::build_context(ir, lib);

    auto fused = fusion::group_nodes(ir, ctx);
    fusion::split_at_sampler2d_sources(fused, ctx);
    fusion::merge_groups(fused, ctx);

    std::cout << "\n=== LevelsFanOut: after group_nodes+merge ===\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n])
                      << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
    }

    // Dump cross-group connections before compute_external_inputs
    std::cout << "\n  Cross-group connections:\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        for (NodeId dst : fused.groups[gi].nodes) {
            auto it = ctx.conns_by_dst.find(dst);
            if (it == ctx.conns_by_dst.end()) continue;
            for (const auto& [src_node, src_socket, dst_socket] : it->second) {
                if (!group_contains(fused.groups[gi], src_node)) {
                    auto tit = ctx.node_type.find(dst);
                    if (tit == ctx.node_type.end()) continue;
                    const auto* type = tit->second;
                    std::cout << "    " << node_name(src_node) << "(" << src_node
                              << ") -> " << node_name(dst) << "(" << dst
                              << ") socket=" << dst_socket
                              << " type=" << (type->inputs[dst_socket].type == SocketType::Sampler2D ? "Sampler2D" : "Vec4")
                              << "\n";
                }
            }
        }
    }

    fusion::compute_external_inputs(fused, ctx);

    std::cout << "\n  After compute_external_inputs:\n";
    for (size_t gi = 0; gi < fused.groups.size(); ++gi) {
        std::cout << "  Group " << gi << ": [";
        for (size_t n = 0; n < fused.groups[gi].nodes.size(); ++n) {
            if (n) std::cout << ", ";
            std::cout << node_name(fused.groups[gi].nodes[n])
                      << "(" << fused.groups[gi].nodes[n] << ")";
        }
        std::cout << "]\n";
        for (const auto& ext : fused.groups[gi].external_inputs) {
            std::cout << "    ext: " << node_name(ext.src_node)
                      << "(" << ext.src_node << ") -> "
                      << node_name(ext.dst_node)
                      << "(" << ext.dst_node << ") slot=" << ext.slot << "\n";
        }
    }

    auto compiled = fusion::compile_groups(fused, ir, ctx);
    ASSERT_TRUE(compiled.ok()) << compiled.error;

    // Expect 2 groups: [Perlin,Levels] and [Warp,Invert,Blend]
    EXPECT_EQ(fused.groups.size(), 2u);

    // Levels(2)→Invert(4) is Vec4 cross-group — this is the bug test
    bool found_levels_invert_ext = false;
    for (auto& fg : fused.groups)
        for (auto& ext : fg.external_inputs)
            if (ext.src_node == 2 && ext.dst_node == 4)
                found_levels_invert_ext = true;
    EXPECT_TRUE(found_levels_invert_ext)
        << "Levels(2) -> Invert(4) Vec4 cross-group connection must be tracked as external input";

    // Levels(2)→Warp(3) is Sampler2D cross-group
    bool found_levels_warp_ext = false;
    for (auto& fg : fused.groups)
        for (auto& ext : fg.external_inputs)
            if (ext.src_node == 2 && ext.dst_node == 3)
                found_levels_warp_ext = true;
    EXPECT_TRUE(found_levels_warp_ext)
        << "Levels(2) -> Warp(3) Sampler2D cross-group connection must be tracked";

    EXPECT_EQ(compiled.groups.back().output_node, 5u);
}
