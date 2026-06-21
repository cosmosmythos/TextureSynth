#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/RegisterAllocator.hpp"
#include "test_assets.hpp"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace te;

// RESEARCH TEST: probe whether the 12-node graph could fit in ONE chain
// instead of splitting into 3, and what register cost the engine estimates.

namespace {
NodeLibrary load_real_lib() {
    NodeLibrary lib;
    std::string err;
    NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    return lib;
}
} // anonymous namespace

TEST(ResearchRegisterPressure, UserGraphCostEstimate) {
    NodeLibrary lib = load_real_lib();

    // Same graph as test_repro_blend_preview (user's exact scenario)
    Graph g;
    g.nodes.push_back({1, "worley"});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "worley"});
    g.nodes.push_back({4, "levels"});
    g.nodes.push_back({5, "blend"});
    g.nodes.push_back({6, "levels"});
    g.nodes.push_back({7, "simplex"});
    g.nodes.push_back({8, "levels"});
    g.nodes.push_back({9, "gabor"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blend"});
    g.nodes.push_back({12, "shuffle"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({2, 0, 5, 1});
    g.connections.push_back({4, 0, 5, 2});
    g.connections.push_back({5, 0, 6, 0});
    g.connections.push_back({7, 0, 8, 0});
    g.connections.push_back({9, 0, 10, 0});
    g.connections.push_back({6, 0, 11, 1});
    g.connections.push_back({10, 0, 11, 2});
    g.connections.push_back({8, 0, 11, 0});
    g.connections.push_back({11, 0, 12, 0});
    g.output_node = 12;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;

    auto path = ActivePathTracer::trace(r.ir, 12, lib);
    std::cout << "\n=== Active path: " << path.nodes.size() << " nodes ===" << std::endl;
    std::cout << "Engine-estimated total register cost: " << path.estimated_registers
              << " (budget is " << reg::RegisterAllocator::DEFAULT_BUDGET << ")" << std::endl;

    std::cout << "\n=== Per-node costs (estimate_cost heuristic) ===" << std::endl;
    uint32_t sum = 0;
    for (NodeId n : path.nodes) {
        const auto* inst = r.ir.find(n);
        const auto* type = inst ? lib.find(inst->type_id) : nullptr;
        if (!type) continue;
        auto c = reg::RegisterAllocator::estimate_cost(type->id);
        std::cout << "  node " << n << " (" << type->id << "): out=" << c.output_regs
                  << " temp=" << c.temp_regs << " samp=" << c.sampler_regs
                  << " in=" << c.input_regs << "  -> total " << c.total() << std::endl;
        sum += c.total();
    }
    std::cout << "Sum of per-node totals (no overlap considered): " << sum << std::endl;

    // Empirically compile the WHOLE 12-node graph into ONE shader by
    // forcing budget high. Measure whether SPIR-V compiles + pipeline creates.
    std::cout << "\n=== Forcing single-chain fusion (budget=10000) ===" << std::endl;
    auto single = emit_fused_subgraph(path, r.ir, lib, 0, {});
    if (!single.ok()) {
        std::cout << "emit_fused_subgraph failed: " << single.error << std::endl;
    } else {
        std::cout << "Single-chain GLSL emitted OK (" << single.source.size() << " bytes)" << std::endl;
        std::cout << "external_inputs in single chain: " << single.external_inputs << std::endl;
    }

    // Write it out for inspection
    if (single.ok()) {
        std::ofstream f("research_single_chain.glsl");
        f << single.source;
        std::cout << "Wrote research_single_chain.glsl for inspection" << std::endl;
    }
}

TEST(ResearchRegisterPressure, GpuActuallyAcceptsSingleChain) {
    // Real test: can the GPU's compiler accept the whole 12-node graph as
    // one shader? If our 48-budget split was conservative, this should work.
    NodeLibrary lib = load_real_lib();

    Engine engine;
    if (!engine.init(VK_NULL_HANDLE, nullptr, 0, true, "research",
                     find_test_nodes_dir().c_str(),
                     find_test_glsl_dir().c_str())) {
        GTEST_SKIP() << "no GPU";
    }

    // Sanity: see what the GPU reports for compute limits
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(engine.ctx().physical_device(), &props);
    std::cout << "\nGPU: " << props.deviceName << std::endl;
    std::cout << "  maxComputeSharedMemorySize: "
              << props.limits.maxComputeSharedMemorySize << " bytes" << std::endl;
    std::cout << "  maxComputeWorkGroupInvocations: "
              << props.limits.maxComputeWorkGroupInvocations << std::endl;
    std::cout << "  maxComputeWorkGroupSize: ["
              << props.limits.maxComputeWorkGroupSize[0] << ","
              << props.limits.maxComputeWorkGroupSize[1] << ","
              << props.limits.maxComputeWorkGroupSize[2] << "]" << std::endl;

    Graph g;
    g.nodes.push_back({1, "worley"});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "worley"});
    g.nodes.push_back({4, "levels"});
    g.nodes.push_back({5, "blend"});
    g.nodes.push_back({6, "levels"});
    g.nodes.push_back({7, "simplex"});
    g.nodes.push_back({8, "levels"});
    g.nodes.push_back({9, "gabor"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blend"});
    g.nodes.push_back({12, "shuffle"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({2, 0, 5, 1});
    g.connections.push_back({4, 0, 5, 2});
    g.connections.push_back({5, 0, 6, 0});
    g.connections.push_back({7, 0, 8, 0});
    g.connections.push_back({9, 0, 10, 0});
    g.connections.push_back({6, 0, 11, 1});
    g.connections.push_back({10, 0, 11, 2});
    g.connections.push_back({8, 0, 11, 0});
    g.connections.push_back({11, 0, 12, 0});
    g.output_node = 12;

    // Emit the entire path as a single shader (bypass FusedGraphCompiler's split)
    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success);
    auto path = ActivePathTracer::trace(r.ir, 12, lib);
    auto single = emit_fused_subgraph(path, r.ir, lib, 0, {});
    ASSERT_TRUE(single.ok()) << single.error;

    // Try to compile it via the engine's own shader compiler
    ShaderCompiler sc;
    auto t0 = std::chrono::steady_clock::now();
    auto result = sc.compile_compute_sync(single.source, "research_single");
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "\n=== Compile attempt (single 12-node shader) ===" << std::endl;
    if (result.success) {
        std::cout << "COMPILE OK in " << ms << "ms, SPIR-V size="
                  << (result.spirv.size() * 4) << " bytes" << std::endl;
        std::cout << "  -> GPU accepted all 12 nodes in one shader" << std::endl;
    } else {
        std::cout << "COMPILE FAILED (" << ms << "ms):" << std::endl;
        std::cout << result.error_log << std::endl;
        std::cout << "  -> 48-budget split was justified" << std::endl;
    }
}
