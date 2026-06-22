#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "test_assets.hpp"
#include <iostream>
#include <fstream>

using namespace te;

namespace {
NodeLibrary load_lib() {
    NodeLibrary lib;
    std::string err;
    NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    return lib;
}
}

// Same graph as the failing Python test:
//   simplex(1) -> blend.a (socket 1)
//   value(2)   -> blend.b (socket 2)
//   output = blend(3)
TEST(FusedBlendGLSL, NoiseToBlend) {
    NodeLibrary lib = load_lib();
    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "value"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});  // simplex -> blend.a
    g.connections.push_back({2, 0, 3, 2});  // value   -> blend.b
    g.output_node = 3;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto cr = FusedGraphCompiler::compile(r.ir, lib, 3);
    ASSERT_TRUE(cr.success) << cr.error;

    std::cout << "\n=== param_base_slot map ===" << std::endl;
    for (const auto& kv : cr.param_base_slot) {
        const auto* vn = r.ir.find(kv.first);
        const auto* type = vn ? lib.find(vn->type_id) : nullptr;
        std::cout << "  node " << kv.first << " (" << (type ? type->id : "?") << ")"
                  << " base=" << kv.second << std::endl;
        if (type) {
            for (size_t i = 0; i < type->params.size(); ++i)
                std::cout << "    param[" << i << "] " << type->params[i].name
                          << " slot=" << (kv.second + i) << std::endl;
            uint32_t fi = 0;
            for (size_t i = 0; i < type->inputs.size(); ++i) {
                if (type->inputs[i].type == SocketType::Float) {
                    std::cout << "    float_input[" << fi << "] " << type->inputs[i].name
                              << " slot=" << (kv.second + type->params.size() + fi) << std::endl;
                    ++fi;
                }
            }
        }
    }
    std::cout << "total_param_floats=" << cr.total_param_floats << std::endl;

    std::cout << "\n=== Chains (" << cr.pass_plan.chains.size() << ") ===" << std::endl;
    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        std::cout << "\nChain " << ci << " nodes=[";
        for (size_t ni = 0; ni < ch.nodes.size(); ++ni) {
            if (ni) std::cout << ", ";
            const auto* vn = r.ir.find(ch.nodes[ni]);
            const auto* type = vn ? lib.find(vn->type_id) : nullptr;
            std::cout << ch.nodes[ni] << ":" << (type ? type->id : "?");
        }
        std::cout << "]" << std::endl;
        std::cout << "  param_base_slot=" << ch.param_base_slot
                  << " total_params=" << ch.total_params << std::endl;
        std::cout << "  param_offsets=[";
        for (size_t i = 0; i < ch.param_offsets.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << ch.param_offsets[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  param_global_slots=[";
        for (size_t i = 0; i < ch.param_global_slots.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << ch.param_global_slots[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  external_socket_masks=[";
        for (size_t i = 0; i < ch.external_socket_masks.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << ch.external_socket_masks[i];
        }
        std::cout << "]" << std::endl;

        // Dump GLSL
        std::cout << "\n--- Chain " << ci << " GLSL ---\n" << ch.glsl << std::endl;
    }
}
