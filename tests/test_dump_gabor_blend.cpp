#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "test_assets.hpp"
#include <fstream>
#include <iostream>

using namespace te;

namespace {
NodeLibrary load_real_lib() {
    NodeLibrary lib;
    std::string err;
    int n = NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    EXPECT_GT(n, 0) << "failed to load real nodes: " << err;
    return lib;
}
} // anonymous namespace

TEST(DumpGaborBlend, ExactUserGraph) {
    NodeLibrary lib = load_real_lib();

    Graph g;
    g.nodes.push_back({5199789026961869479, "gabor"});
    g.nodes.push_back({12915422993863950465, "gabor"});
    g.nodes.push_back({2105540117126596610, "worley"});
    g.nodes.push_back({1475714937147705185, "remap"});
    g.nodes.push_back({1999265074042586141, "blend"});

    // Gabor1 -> Blend.A (socket 1)
    g.connections.push_back({5199789026961869479, 0, 1999265074042586141, 1});
    // Gabor2 -> Blend.B (socket 2)
    g.connections.push_back({12915422993863950465, 0, 1999265074042586141, 2});
    // Worley -> Remap (socket 0)
    g.connections.push_back({2105540117126596610, 0, 1475714937147705185, 0});
    // Remap -> Blend.Mask (socket 0)
    g.connections.push_back({1475714937147705185, 0, 1999265074042586141, 0});

    g.output_node = 1999265074042586141;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto path = ActivePathTracer::trace(r.ir, g.output_node, lib);
    std::cout << "=== Active path ===" << std::endl;
    for (NodeId n : path.nodes) {
        const auto* inst = r.ir.find(n);
        const auto* type = inst ? lib.find(inst->type_id) : nullptr;
        std::cout << "  " << n << " -> " << (type ? type->id : "?") << std::endl;
    }

    auto cr = FusedGraphCompiler::compile(r.ir, lib, g.output_node);
    ASSERT_TRUE(cr.success) << cr.error;

    for (size_t ci = 0; ci < cr.pass_plan.chains.size(); ++ci) {
        const auto& ch = cr.pass_plan.chains[ci];
        std::cout << "\n=== Chain " << ci << " (nodes=" << ch.nodes.size()
                  << ", total_params=" << ch.total_params
                  << ", param_base_slot=" << ch.param_base_slot << ") ===" << std::endl;
        std::cout << "Nodes: ";
        for (NodeId n : ch.nodes) std::cout << n << " ";
        std::cout << "\nParam offsets: ";
        for (uint32_t o : ch.param_offsets) std::cout << o << " ";
        std::cout << "\nParam global slots: ";
        for (uint32_t s : ch.param_global_slots) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << ch.glsl << std::endl;
    }
}
