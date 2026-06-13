#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "engine/Engine.hpp"
#include <iostream>

using namespace ts;

// Test the exact scenario: simplex → invert(muted) → blend(active)
// Expected: blend should read from simplex (after rewire), output = blend(simplex, unlinked, unlinked)
TEST(MuteMiddleNodeDebug, SimplexMutedInvertBlend) {
    NodeLibrary lib;
    lib.load_json_manifest("shader_assets/node_manifest.json");

    Graph graph;
    NodeId simplex_id = NodeId(1);
    NodeId invert_id = NodeId(2);
    NodeId blend_id = NodeId(3);

    // Add nodes matching your Blender setup
    graph.add_node(simplex_id, "Simplex", 0, "Simplex");
    graph.add_node(invert_id, "Invert", 0, "Invert", /*muted=*/true);
    graph.add_node(blend_id, "Blend", 0, "Blend", /*muted=*/false);

    // Add connections: simplex(0) → invert(0), invert(0) → blend(0)
    graph.add_connection(simplex_id, 0, invert_id, 0);
    graph.add_connection(invert_id, 0, blend_id, 0);

    // Set output to blend (the active node)
    graph.output_node = blend_id;

    // Validate
    auto ir_result = validate_graph(graph, lib);
    EXPECT_TRUE(ir_result.success) << "Validation failed: " << ir_result.error;

    // Debug output
    std::cout << "\n=== VALIDATION RESULT ===\n";
    std::cout << "IR output_node: " << static_cast<uint32_t>(ir_result.ir.output_node) << "\n";
    std::cout << "IR nodes count: " << ir_result.ir.nodes.size() << "\n";
    for (const auto& node : ir_result.ir.nodes) {
        std::cout << "  Node " << static_cast<uint32_t>(node.id) << ": " << node.type << "\n";
    }
    std::cout << "IR connections count: " << ir_result.ir.connections.size() << "\n";
    for (const auto& conn : ir_result.ir.connections) {
        std::cout << "  " << static_cast<uint32_t>(conn.src_node) << "[" << conn.src_socket 
                  << "] → " << static_cast<uint32_t>(conn.dst_node) << "[" << conn.dst_socket << "]\n";
    }

    // Expected after rewire:
    // IR should contain: simplex (id=1), blend (id=3)
    // NOT contain: invert (id=2)
    // Connection should be: simplex[0] → blend[0]
    
    EXPECT_EQ(ir_result.ir.nodes.size(), 2) << "Should have 2 nodes (simplex + blend, no invert)";
    EXPECT_EQ(ir_result.ir.output_node, blend_id) << "Output should still be blend";
    
    bool has_simplex_to_blend = false;
    for (const auto& conn : ir_result.ir.connections) {
        if (conn.src_node == simplex_id && conn.dst_node == blend_id && 
            conn.src_socket == 0 && conn.dst_socket == 0) {
            has_simplex_to_blend = true;
        }
    }
    EXPECT_TRUE(has_simplex_to_blend) << "Should have connection simplex[0] → blend[0]";

    // Try to compile
    auto compile_result = FusedGraphCompiler::compile(ir_result.ir, lib, ir_result.ir.output_node);
    EXPECT_TRUE(compile_result.success) << "Compilation failed: " << compile_result.error;
    
    if (compile_result.success) {
        std::cout << "\n=== COMPILATION RESULT ===\n";
        std::cout << "Passes: " << compile_result.compute_passes.size() << "\n";
        for (size_t i = 0; i < compile_result.compute_passes.size(); ++i) {
            std::cout << "  Pass " << i << ": " << (compile_result.compute_passes[i].shader_glsl.empty() ? "EMPTY" : "OK") << "\n";
        }
    }
}
