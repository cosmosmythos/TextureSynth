#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/ResourceManager.hpp"
#include "engine/Engine.hpp"
#include "test_assets.hpp"
#include <chrono>
#include <thread>
#include <unordered_set>

using namespace te;

// ===========================================================================
// Stage 6: Aliasing — lifetime, color-class, and pool tests.
// ===========================================================================

namespace {

NodeType make_pure_pixel(const std::string& id, uint32_t n_in = 1) {
    NodeType t;
    t.id = id;
    t.display_name = id;
    t.pass_kind = PassKind::PurePixel;
    for (uint32_t i = 0; i < n_in; ++i) {
        Socket s; s.name = "in" + std::to_string(i); s.type = SocketType::Vec4;
        t.inputs.push_back(s);
    }
    Socket out; out.name = "out"; out.type = SocketType::Vec4;
    t.outputs.push_back(out);
    return t;
}

NodeLibrary make_aliasing_lib() {
    NodeLibrary lib;
    lib.add_public(make_pure_pixel("source", 0));   // id 1
    lib.add_public(make_pure_pixel("step",   1));   // id 2
    lib.add_public(make_pure_pixel("fanin",  2));   // id 3 — two-input merge
    return lib;
}

// Build a GraphIR directly (bypass validator pruning) with a known topology
// where two early branches' resources have non-overlapping lifetimes.
//
// Topology (5 nodes):
//   1 (source) -> 2 (step)         R_1: [0,1], R_2: [1,4]
//   3 (source) -> 4 (step)         R_3: [2,3], R_4: [3,4]
//   2, 4 -> 5 (fanin) = output     R_5: pinned
//
// Non-overlapping: R_1 [0,1] and R_3 [2,3] -> share color class.
GraphIR make_aliasing_ir() {
    GraphIR ir;
    auto add = [&](NodeId id, const std::string& type, size_t idx) {
        ValidatedNode n;
        n.id = id; n.type_id = type;
        n.debug_name = type + "_" + std::to_string(id);
        ir.nodes.push_back(n);
        ir.node_index[id] = idx;
    };
    add(1, "source", 0);
    add(2, "step",   1);
    add(3, "source", 2);
    add(4, "step",   3);
    add(5, "fanin",  4);
    // Connections: 1->2.s0, 3->4.s0, 2->5.s0, 4->5.s1
    ir.connections.push_back({1, 0, 2, 0});
    ir.connections.push_back({3, 0, 4, 0});
    ir.connections.push_back({2, 0, 5, 0});
    ir.connections.push_back({4, 0, 5, 1});
    // Topological eval order
    ir.eval_order = {1, 3, 2, 4, 5};
    ir.output_node = 5;
    return ir;
}

} // namespace

// ---------------------------------------------------------------------------
// Unit: lifetimes and chain_index_of_pass populated correctly.
// ---------------------------------------------------------------------------
TEST(Aliasing, LifetimesAndChainIndexArePopulated) {
    NodeLibrary lib = make_aliasing_lib();
    GraphIR ir = make_aliasing_ir();

    auto cr = GraphCompiler::compile(ir, lib);
    ASSERT_TRUE(cr.success) << cr.error;
    const PassPlan& plan = cr.pass_plan;

    // 5 passes; chain_index_of_pass same size.
    ASSERT_EQ(plan.passes.size(), 5u);
    ASSERT_EQ(plan.chain_index_of_pass.size(), 5u);

    // Lifetimes for every output resource must have a finite first_pass and last_pass.
    for (size_t i = 0; i < plan.passes.size(); ++i) {
        for (auto& rid : plan.passes[i].output_resources) {
            auto lt = plan.lifetimes.find(rid);
            ASSERT_NE(lt, plan.lifetimes.end()) << "missing lifetime for rid " << rid.node_id;
            EXPECT_NE(lt->second.first_pass, UINT32_MAX);
            EXPECT_GE(lt->second.last_pass, lt->second.first_pass);
        }
    }

    // final_output_resource is pinned (last_pass = UINT32_MAX).
    auto fo = plan.lifetimes.find(plan.final_output_resource);
    ASSERT_NE(fo, plan.lifetimes.end());
    EXPECT_EQ(fo->second.first_pass, 0u);
    EXPECT_EQ(fo->second.last_pass, UINT32_MAX);
}

// ---------------------------------------------------------------------------
// Unit: non-overlapping lifetimes land in the same color class.
// ---------------------------------------------------------------------------
TEST(Aliasing, NonOverlappingResourcesShareColor) {
    NodeLibrary lib = make_aliasing_lib();
    GraphIR ir = make_aliasing_ir();

    auto cr = GraphCompiler::compile(ir, lib);
    ASSERT_TRUE(cr.success) << cr.error;
    const PassPlan& plan = cr.pass_plan;

    // With eval_order={1,3,2,4,5}, lifetimes are:
    //   R_1 [0,2], R_3 [1,3], R_2 [2,4], R_4 [3,4]
    // R_1.last=2 < R_4.first=3 -> non-overlapping -> same color.
    auto c1 = plan.color_classes.find({1, 0});
    auto c4 = plan.color_classes.find({4, 0});
    ASSERT_NE(c1, plan.color_classes.end());
    ASSERT_NE(c4, plan.color_classes.end());
    EXPECT_EQ(c1->second, c4->second)
        << "R_1 and R_4 have non-overlapping lifetimes (R_1.last=2 < R_4.first=3) — must share a color class";

    // R_2 and R_4 both end at the final pass (pass 4) — NOT aliased.
    auto c2 = plan.color_classes.find({2, 0});
    ASSERT_NE(c2, plan.color_classes.end());
    EXPECT_NE(c2->second, c4->second)
        << "R_2 and R_4 both end at the final pass — must NOT share a color class";

    // Pinned (color 0): final output.
    auto c5 = plan.color_classes.find({5, 0});
    if (c5 != plan.color_classes.end()) {
        EXPECT_EQ(c5->second, 0u) << "pinned resource must have color 0";
    }
}

// ---------------------------------------------------------------------------
// Unit: overlapping resources get different colors.
// ---------------------------------------------------------------------------
TEST(Aliasing, LinearChainProducesNoAliasing) {
    NodeLibrary lib = make_aliasing_lib();
    GraphIR ir;
    auto add = [&](NodeId id, const std::string& type, size_t idx) {
        ValidatedNode n;
        n.id = id; n.type_id = type;
        n.debug_name = type + "_" + std::to_string(id);
        ir.nodes.push_back(n);
        ir.node_index[id] = idx;
    };
    add(1, "source", 0);
    add(2, "step",   1);
    add(3, "step",   2);
    ir.connections.push_back({1, 0, 2, 0});
    ir.connections.push_back({2, 0, 3, 0});
    ir.eval_order = {1, 2, 3};
    ir.output_node = 3;

    auto cr = GraphCompiler::compile(ir, lib);
    ASSERT_TRUE(cr.success);
    // In a linear chain, every adjacent pair overlaps at one pass -> no two should alias.
    auto c1 = cr.pass_plan.color_classes.find({1, 0});
    auto c2 = cr.pass_plan.color_classes.find({2, 0});
    ASSERT_NE(c1, cr.pass_plan.color_classes.end());
    ASSERT_NE(c2, cr.pass_plan.color_classes.end());
    EXPECT_NE(c1->second, c2->second);
}

// ---------------------------------------------------------------------------
// Integration: real engine + real graph + VMA stats. Verify aliasing
// actually causes VMA physical bytes < logical bytes.
// ---------------------------------------------------------------------------
TEST(Aliasing, EngineAllocatesAliasPoolsAndReducesVmaBytes) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }

    // Build the diamond: perlin -> invert,  perlin -> invert,  blend(output)
    Graph g;
    g.nodes.push_back({1, "perlin",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({2, "invert",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({3, "perlin",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({4, "invert",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({5, "blend",    ChannelFormat::RGBA, "", false, false});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({2, 0, 5, 0});
    g.connections.push_back({4, 0, 5, 1});
    g.output_node = 5;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << "set_graph failed: " << engine.last_error();

    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    // 5 output resources, 512x512 each. With the diamond graph the
    // validator's Kahn-order eval_order = {1,2,3,4,5}, so lifetimes are
    // R_1 [0,1], R_2 [1,4], R_3 [2,3], R_4 [3,4], R_5 pinned.
    // R_1 and R_3 are non-overlapping -> share a color class -> one shared
    // VmaAllocation. The other intermediates are singletons.
    auto stats = engine.resources().get_vma_stats(engine.ctx());
    EXPECT_EQ(stats.node_resource_count, 5u);
    EXPECT_GT(stats.node_resource_bytes, 0u);

    // Diagnostic: print per-heap stats so failures show the breakdown.
    std::cerr << "[stats] node_resource_bytes=" << stats.node_resource_bytes
              << " device_local_alloc=" << stats.device_local_allocation_bytes
              << " vma_allocation_bytes=" << stats.vma_allocation_bytes
              << " vma_block_bytes=" << stats.vma_block_bytes
              << " gpu_usage=" << stats.gpu_usage_bytes
              << " gpu_budget=" << stats.gpu_budget_bytes << "\n";
    for (auto& h : stats.heap_stats) {
        std::cerr << "[heap " << h.index << " " << h.label
                  << "] alloc_bytes=" << h.vma_allocation_bytes
                  << " block_count=" << h.vma_block_count
                  << " alloc_count=" << h.vma_allocation_count
                  << " heap_size=" << h.heap_size_bytes << "\n";
    }

    // VMA device-local allocation bytes should be strictly less than the
    // sum of logical image bytes. The diamond's 5 images sum to 10 MB;
    // aliasing R_1/R_3 into a single VmaAllocation drops the graph to 8 MB
    // of device-local VmaAllocation bytes. The output_storage_ (4 MB,
    // owned by Engine, not ResourceManager) is not counted in
    // node_resource_bytes but does live in the same device-local heap --
    // so the comparison is graph + output_storage vs the device-local
    // total. With 4 graph VmaAllocs (1 shared + 3 pinned) and 1
    // output_storage, total device-local should be ~12 MB for a 10 MB
    // graph (savings visible) or ~14 MB without aliasing.
    EXPECT_LT(stats.device_local_allocation_bytes, stats.node_resource_bytes + 4u * 1024u * 1024u)
        << "aliasing should reduce device-local VMA bytes below logical sum "
           "(saving ~1 image worth = 2 MB)";

    // Render to make sure the aliasing doesn't break correctness.
    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);

    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200 && !engine.async_readback().poll(engine.ctx(), pixels, w, h, og); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(w, 512u);
    EXPECT_EQ(h, 512u);
    EXPECT_EQ(pixels.size(), 512u * 512u * 4u);
    // Sanity: pixels are not all zero (a real render happened).
    double sum = 0.0;
    for (size_t i = 0; i < pixels.size(); i += 4) sum += pixels[i];
    EXPECT_GT(sum, 0.0) << "rendered image is all zeros — aliasing corrupted output";

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// Integration: single-output graph (no aliasing possible) still renders
// correctly with alias_group_id == 0 on every resource.
// ---------------------------------------------------------------------------
TEST(Aliasing, SingleNodeNoAliasingStillRenders) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_single",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "", false, false});
    g.output_node = 1;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    // Single node = single output resource, no aliasing.
    auto stats = engine.resources().get_vma_stats(engine.ctx());
    EXPECT_EQ(stats.node_resource_count, 1u);
    // aliasing_efficiency >= 1.0 (no savings, possibly alignment waste).
    double eff = stats.aliasing_efficiency();
    EXPECT_GE(eff, 1.0 - 0.01) << "single-resource graph: efficiency should be ~1.0";

    // Render still works.
    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200 && !engine.async_readback().poll(engine.ctx(), pixels, w, h, og); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_GT(pixels.size(), 0u);
    engine.shutdown();
}
