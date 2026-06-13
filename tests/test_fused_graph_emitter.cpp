#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"

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

TEST(FusedGraphEmitter, LinearChain) {
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
    auto result = emit_fused_subgraph(path, r.ir, lib, 0);

    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_FALSE(result.source.empty());
    EXPECT_EQ(result.external_inputs, 0u);

    EXPECT_NE(result.source.find("node_source"), std::string::npos);
    EXPECT_NE(result.source.find("node_step"), std::string::npos);
    EXPECT_NE(result.source.find("_local_0"), std::string::npos);
    EXPECT_NE(result.source.find("_local_2"), std::string::npos);
    EXPECT_NE(result.source.find("imageStore"), std::string::npos);
}

TEST(FusedGraphEmitter, ExternalInputs) {
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

    // Select blend node (socket 1 is unconnected Vec4 = baked constant)
    auto path = ActivePathTracer::trace(r.ir, 3, lib);
    auto result = emit_fused_subgraph(path, r.ir, lib, 0);

    ASSERT_TRUE(result.ok()) << result.error;
    // Unconnected Vec4 is baked as vec4(0.0) — no external slot consumed
    EXPECT_EQ(result.external_inputs, 0u);
    EXPECT_EQ(result.source.find("texelFetch"), std::string::npos)
        << "unconnected Vec4 must not use texelFetch";
    // Must contain baked constant
    EXPECT_NE(result.source.find("vec4(0"), std::string::npos)
        << "unconnected Vec4 must be baked as vec4(0.0)";
}

TEST(FusedGraphEmitter, EmptyPathReturnsError) {
    auto lib = make_lib();
    ActivePath empty_path;
    GraphIR ir;
    auto result = emit_fused_subgraph(empty_path, ir, lib, 0);
    EXPECT_FALSE(result.ok());
}

TEST(FusedGraphEmitter, ParamsEmitted) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step1p"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto path = ActivePathTracer::trace(r.ir, 2, lib);
    auto result = emit_fused_subgraph(path, r.ir, lib, 5);

    ASSERT_TRUE(result.ok()) << result.error;
    EXPECT_NE(result.source.find("node_params"), std::string::npos);
    EXPECT_NE(result.source.find("pc.param_base_slot"), std::string::npos);
}

TEST(FusedGraphEmitter, FloatInputReadsSSBO) {
    // Simulate blend-like node: 2 vec4 inputs + 1 float input + 1 float param.
    // The float input (mask) is unconnected — should read from SSBO, not hardcoded 1.0.
    auto lib = make_lib();

    NodeType blend_ft;
    blend_ft.id = "blend_ft";
    blend_ft.display_name = "blend_ft";
    blend_ft.pass_kind = PassKind::Compute;
    {
        Socket s; s.name = "mask"; s.type = SocketType::Float;
        blend_ft.inputs.push_back(s);
    }
    {
        Socket s; s.name = "a"; s.type = SocketType::Vec4;
        blend_ft.inputs.push_back(s);
    }
    {
        Socket s; s.name = "b"; s.type = SocketType::Vec4;
        blend_ft.inputs.push_back(s);
    }
    {
        Socket s; s.name = "out"; s.type = SocketType::Vec4;
        blend_ft.outputs.push_back(s);
    }
    {
        NodeParam p; p.name = "mode"; blend_ft.params.push_back(p);
    }
    blend_ft.glsl_function =
        "vec4 node_blend_ft(vec2 uv, float mask, vec4 a, vec4 b, float mode)"
        " { return mix(a, b, mask); }";
    lib.add_public(std::move(blend_ft));

    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "blend_ft"});
    g.connections.push_back({1, 0, 2, 1});   // source -> blend_ft.a (socket 1)
    // mask (socket 0) and b (socket 2) are unconnected
    g.output_node = 2;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto path = ActivePathTracer::trace(r.ir, 2, lib);
    auto result = emit_fused_subgraph(path, r.ir, lib, 0);

    ASSERT_TRUE(result.ok()) << result.error;

    // Float input (mask) should read from SSBO, NOT use literal 1.0.
    EXPECT_NE(result.source.find("node_params"), std::string::npos)
        << "unconnected float input should read from param SSBO";
    // Should NOT contain the old hardcoded "1.0" as a standalone literal for mask.
    // The SSBO read pattern: node_params[pc.param_ring_idx].v[pc.param_base_slot + ...]
    EXPECT_NE(result.source.find("pc.param_base_slot"), std::string::npos);
    // 1 manifest param (mode) + 1 float input (mask) = 2 SSBO slots consumed.
    // The mask default is at param_base_slot + 1 (after mode).
}
