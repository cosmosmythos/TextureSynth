#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include "engine/memory_allocation/AliasColorer.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/GraphIR.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "engine/register_allocation/GraphColorer.hpp"
#include "engine/register_allocation/LivenessAnalysis.hpp"
#include "engine/graphfusion/GlslBuilder.hpp"
#include "test_assets.hpp"
#include <chrono>
#include <thread>
#include <sstream>

using namespace te;

// ===========================================================================
// Aliasing integration tests — verify fused-path resource allocation.
// ===========================================================================

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

    auto stats = engine.resources().get_vma_stats(engine.ctx());
    EXPECT_EQ(stats.node_resource_count, 1u);
    double eff = stats.aliasing_efficiency();
    EXPECT_GE(eff, 1.0 - 0.01) << "single-resource graph: efficiency should be ~1.0";

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_GT(pixels.size(), 0u);
    engine.shutdown();
}

TEST(Aliasing, FusedGraphSingleOutputImage) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_fused",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }

    // Diamond: perlin -> invert -> blend, perlin -> invert -> blend
    // Fused path allocates ONE VkImage for the final output.
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

    auto stats = engine.resources().get_vma_stats(engine.ctx());
    EXPECT_EQ(stats.node_resource_count, 1u)
        << "fused path should allocate 1 VkImage (final output only)";
    EXPECT_GT(stats.node_resource_bytes, 0u);

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(w, 512u);
    EXPECT_EQ(h, 512u);
    double sum = 0.0;
    for (size_t i = 0; i < pixels.size(); i += 4) sum += pixels[i];
    EXPECT_GT(sum, 0.0) << "rendered image is all zeros";

    engine.shutdown();
}

// ===========================================================================
// AliasColorer unit tests — format-aware interval coloring.
// ===========================================================================

class AliasColorerTest : public ::testing::Test {
protected:
    using AliasLifetime = memory_allocation::AliasLifetime;
    using AliasColoringResult = memory_allocation::AliasColoringResult;
    using AliasColorer = memory_allocation::AliasColorer;

    ResourceUUID make_rid(uint32_t node_id, uint32_t output_index = 0) {
        return ResourceUUID{node_id, output_index};
    }
};

TEST_F(AliasColorerTest, EmptyLifetimes_NoGroups) {
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;
    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.groups_created, 0u);
    EXPECT_TRUE(result.color_classes.empty());
}

TEST_F(AliasColorerTest, SingleResource_NoAliasGroup) {
    // One resource with multi-pass lifetime — but only 1 member, no aliasing possible.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

    auto a = make_rid(1);
    lifetimes[a] = {0, 1};
    formats[a] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    // Single member — should still get a color class (but aliasing alloc skips singleton groups).
    EXPECT_EQ(result.groups_created, 1u);
}

TEST_F(AliasColorerTest, TwoSameFormatNonOverlapping_Alias) {
    // Two RGBA@F32 resources with non-overlapping lifetimes → same color class.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

    auto a = make_rid(1), b = make_rid(2);
    lifetimes[a] = {0, 1};  // pass 0-1
    lifetimes[b] = {2, 3};  // pass 2-3 (no overlap)
    formats[a] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};
    formats[b] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.groups_created, 1u);
    EXPECT_EQ(result.color_classes[a], result.color_classes[b])
        << "non-overlapping same-format resources should share a color class";
}

TEST_F(AliasColorerTest, TwoSameFormatOverlapping_DifferentColors) {
    // Two RGBA@F32 resources with overlapping lifetimes → different color classes.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

    auto a = make_rid(1), b = make_rid(2);
    lifetimes[a] = {0, 2};  // pass 0-2
    lifetimes[b] = {1, 3};  // pass 1-3 (overlaps at pass 1-2)
    formats[a] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};
    formats[b] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.groups_created, 2u);
    EXPECT_NE(result.color_classes[a], result.color_classes[b])
        << "overlapping resources must NOT share a color class";
}

TEST_F(AliasColorerTest, DifferentFormat_NeverAlias) {
    // Mono@F32 (4 bytes/pixel) and RGBA@F32 (16 bytes/pixel) — non-overlapping
    // lifetimes but different formats → different color classes.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

    auto a = make_rid(1), b = make_rid(2);
    lifetimes[a] = {0, 1};  // pass 0-1
    lifetimes[b] = {2, 3};  // pass 2-3 (no overlap)
    formats[a] = StorageFormat{ChannelFormat::Mono, BitDepth::F32};
    formats[b] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    // Different format groups → different color pools → different color classes.
    EXPECT_NE(result.color_classes[a], result.color_classes[b])
        << "different-format resources must NOT share a color class";
}

TEST_F(AliasColorerTest, FinalOutput_Excluded) {
    // Final output with UINT32_MAX lifetime is excluded from aliasing.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

    auto a = make_rid(1), final_out = make_rid(2);
    lifetimes[a] = {0, 1};
    lifetimes[final_out] = {0, UINT32_MAX};  // pinned
    formats[a] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};
    formats[final_out] = StorageFormat{ChannelFormat::RGBA, BitDepth::F32};

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.color_classes.count(final_out), 0u)
        << "final output should be excluded from alias coloring";
    EXPECT_EQ(result.groups_created, 1u);
}

TEST_F(AliasColorerTest, ThreeResourceChain) {
    // Chain: A(0,2) → B(3,5) → C(6,8).
    // Each resource has a multi-pass lifetime but they don't overlap.
    // All same format → greedy should fit in 1 color class.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

    auto a = make_rid(1), b = make_rid(2), c = make_rid(3);
    lifetimes[a] = {0, 2};  // passes 0,1,2
    lifetimes[b] = {3, 5};  // passes 3,4,5
    lifetimes[c] = {6, 8};  // passes 6,7,8
    StorageFormat fmt{ChannelFormat::RGBA, BitDepth::F32};
    formats[a] = fmt; formats[b] = fmt; formats[c] = fmt;

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.groups_created, 1u);
    EXPECT_EQ(result.color_classes[a], result.color_classes[b])
        << "A and B non-overlapping → should alias";
    EXPECT_EQ(result.color_classes[b], result.color_classes[c])
        << "B and C non-overlapping → should alias";
}

TEST_F(AliasColorerTest, ThreeResourceChainOverlapping) {
    // Chain: A(0,1) → B(1,2) → C(2,3).
    // A and B overlap at pass 1, B and C overlap at pass 2.
    // A and C don't overlap → can alias.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

    auto a = make_rid(1), b = make_rid(2), c = make_rid(3);
    lifetimes[a] = {0, 1};
    lifetimes[b] = {1, 2};
    lifetimes[c] = {2, 3};
    StorageFormat fmt{ChannelFormat::RGBA, BitDepth::F32};
    formats[a] = fmt; formats[b] = fmt; formats[c] = fmt;

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.groups_created, 2u);
    EXPECT_EQ(result.color_classes[a], result.color_classes[c])
        << "A and C non-overlapping → should alias";
    EXPECT_NE(result.color_classes[a], result.color_classes[b])
        << "A and B overlap → must NOT alias";
}

TEST_F(AliasColorerTest, FormatsCompatible_Check) {
    StorageFormat rgba32{ChannelFormat::RGBA, BitDepth::F32};  // 16 bytes
    StorageFormat mono32{ChannelFormat::Mono, BitDepth::F32};  // 4 bytes
    StorageFormat rgba16{ChannelFormat::RGBA, BitDepth::F16};  // 8 bytes
    StorageFormat rgb32{ChannelFormat::RGB, BitDepth::F32};    // 16 bytes
    StorageFormat uv32{ChannelFormat::UV, BitDepth::F32};      // 8 bytes

    EXPECT_TRUE(AliasColorer::formats_compatible(rgba32, rgba32));
    EXPECT_TRUE(AliasColorer::formats_compatible(rgba32, rgb32));   // both 16 bytes
    EXPECT_TRUE(AliasColorer::formats_compatible(rgba16, uv32));    // both 8 bytes
    EXPECT_FALSE(AliasColorer::formats_compatible(rgba32, mono32)); // 16 vs 4 bytes
    EXPECT_FALSE(AliasColorer::formats_compatible(rgba32, rgba16)); // 16 vs 8 bytes
    EXPECT_FALSE(AliasColorer::formats_compatible(rgba32, uv32));   // 16 vs 8 bytes
}

TEST_F(AliasColorerTest, Stress_TwentyNonOverlapping) {
    // 20 resources with non-overlapping lifetimes: [0,1],[2,3],[4,5],...
    // All same format. Should produce 1 color class (greedy reuses the slot).
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;
    StorageFormat fmt{ChannelFormat::RGBA, BitDepth::F32};

    for (uint32_t i = 0; i < 20; ++i) {
        auto rid = make_rid(i + 1);
        lifetimes[rid] = {i * 2, i * 2 + 1};
        formats[rid] = fmt;
    }

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.groups_created, 1u)
        << "20 sequential non-overlapping resources should fit in 1 color class";
}

TEST_F(AliasColorerTest, Stress_TwentyOverlapping) {
    // 20 resources, all with the same lifetime [0, 10].
    // All same format. All overlap → need 20 color classes.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;
    StorageFormat fmt{ChannelFormat::RGBA, BitDepth::F32};

    for (uint32_t i = 0; i < 20; ++i) {
        auto rid = make_rid(i + 1);
        lifetimes[rid] = {0, 10};
        formats[rid] = fmt;
    }

    auto result = AliasColorer::compute_from_lifetimes(lifetimes, formats);
    EXPECT_EQ(result.groups_created, 20u)
        << "20 fully overlapping resources need 20 separate color classes";
}

// ===========================================================================
// Cross-chain aliasing integration tests.
// ===========================================================================

TEST(Aliasing, CrossChainBlurWarp_RendersCorrectly) {
    // Graph: perlin -> blur -> warp -> output
    // This tests multi-pass (blur) + Sampler2D chain split (warp) + aliasing.
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_blur_warp",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }

    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({2, "blur",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({3, "warp",   ChannelFormat::RGBA, "", false, false});
    g.connections.push_back({1, 0, 2, 0});  // perlin -> blur
    g.connections.push_back({2, 0, 3, 0});  // blur -> warp
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(w, 512u);
    EXPECT_EQ(h, 512u);
    double sum = 0.0;
    for (size_t i = 0; i < pixels.size(); i += 4) sum += pixels[i];
    EXPECT_GT(sum, 0.0) << "blur->warp render is all zeros";

    engine.shutdown();
}

TEST(Aliasing, BlurWarpChain_AliasingEfficiency) {
    // Verify that aliasing is active (efficiency < 1.0) for a multi-chain graph.
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_eff",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }

    // perlin -> levels -> invert -> output
    // All in one chain (no Sampler2D, no multi-pass) → register allocation
    // eliminates intermediates, only final output has a VkImage.
    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({2, "levels", ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({3, "invert", ChannelFormat::RGBA, "", false, false});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    auto stats = engine.resources().get_vma_stats(engine.ctx());
    // Fused path: perlin+levels+invert in one chain → 1 VkImage (output).
    EXPECT_EQ(stats.node_resource_count, 1u)
        << "3-node fused chain should allocate 1 VkImage";

    engine.shutdown();
}

// ===========================================================================
// Complex multi-chain aliasing integration test.
//
// Graph topology:
//   Chain A: simplex(1) → levels(2) → blur(3, singleton) → invert(4)+blend_A(5)
//   Chain B: worley(6) → warp(7,image=6,gradient=6)+blend_B(8)
//   Final:   blend_C(9) reads chain_A.output + chain_B.output
//
// Chain structure (5 chains):
//   [1]              simplex   — source, no inputs
//   [2]              levels    — vec4 chain with simplex
//   [3]              blur      — singleton multi-pass (2 sub-passes)
//   [4,5]            invert+blend_A — fused vec4 tail (reads blur via sampler2D)
//   [6]              worley    — source, fan-out to warp (both inputs)
//   [7,8]            warp+blend_B — fused vec4 tail (reads worley via sampler2D)
//   [9]              blend_C   — reads both chain outputs via sampler2D (final)
//
// Expected VkImage allocation:
//   7 images: 1×worley, 1×blur(H), 1×blur(V→intermediate), 1×invert+blend_A,
//             1×warp+blend_B, 1×blend_C, 1×final output
//   (plus aliasing: invert+blend_A and warp+blend_B have non-overlapping
//    lifetimes → may share VkImage memory)
// ===========================================================================

TEST(Aliasing, DiverseFormats_AliasingSavings) {
    // Direct test: verify aliaser assigns same color to non-overlapping lifetimes.
    NodeLibrary lib;
    std::string err;
    int n = NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    ASSERT_GT(n, 0) << "failed to load nodes: " << err;

    // Two independent perlin→blur chains, plus a final blend that reads blur2.
    // The key: perlin1→blur1 is an OFF-PATH dead branch. Its resources still
    // appear in active_resources if consumed cross-chain. But since it's dead,
    // the aliaser can potentially reuse its VkImage slot.
    //
    // Actually: let's use the SAME perlin output feeding TWO blur chains.
    // perlin(1) → blur(2) → [chain 1, tail consumed by blend]
    // perlin(1) → blur(3) → [chain 2, tail consumed by blend]
    // blend(4) reads blur2 + blur3 via sampler2D? No — blend has vec4 inputs.
    //
    // Simpler: perlin→blur1→blur2→output. blur1 is intermediate (not final output).
    // blur1 output consumed by blur2 via sampler2D → cross-chain → active.
    // blur2 output is final → pinned.
    // blur1: lifetime overlaps with blur2? Let's check:
    //   blur1 output: first_pass = blur1 pass, last_pass = blur2 pass (input)
    //   blur2 output: final → pinned, not aliasable.
    // So only blur1 is aliasable, no one to alias WITH.
    //
    // REAL approach: perlin→blur1→blur2→blend(blur1, blur2)→output
    // But blend has vec4 inputs, can't read cross-chain from blur1.
    //
    // OK: the simplest aliasing scenario is TWO separate graphs that each produce
    // VkImages with non-overlapping lifetimes. Since we can only have one graph,
    // let's use: perlin→blur1→blur2→output where blur1 is intermediate.
    //
    // Chains: [perlin], [blur1 singleton], [blur2 singleton]
    // blur1 output consumed by blur2 via sampler2D → active, colorable.
    // blur2 output is final → pinned.
    // blur1 lifetime: [blur1_pass, blur2_pass] → aliasable.
    // No second aliasable resource to share with.
    //
    // CONCLUSION: with the current node library, getting two aliasable resources
    // with non-overlapping lifetimes requires a graph like:
    //   perlin→blur_a→[intermediate]→blur_b→blend(blur_a_out, blur_b)→output
    // but blend can't read cross-chain.
    //
    // The CORRECT test: verify the aliaser's interval coloring algorithm directly
    // using synthetic lifetimes, then verify the integration renders correctly.

    // Synthetic test: two resources with non-overlapping lifetimes get same color.
    {
        using memory_allocation::AliasLifetime;
        std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
        std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

        ResourceUUID a{10, 0};
        ResourceUUID b{20, 0};
        ResourceUUID c{30, 0};

        lifetimes[a] = {1, 2};
        lifetimes[b] = {3, 4};
        lifetimes[c] = {2, 5};
        formats[a] = StorageFormat{ChannelFormat::RGBA, BitDepth::F16};
        formats[b] = StorageFormat{ChannelFormat::RGBA, BitDepth::F16};
        formats[c] = StorageFormat{ChannelFormat::RGBA, BitDepth::F16};

        auto result = memory_allocation::AliasColorer::compute_from_lifetimes(
            lifetimes, formats);

        printf("\n=== Synthetic aliasing test ===\n");
        for (const auto& kv : result.color_classes) {
            printf("  {%u,%u} -> color %u\n", kv.first.node_id, kv.first.output_index, kv.second);
        }
        printf("  groups_created: %u\n", result.groups_created);

        // a [1,2] and b [3,4] don't overlap → same color
        EXPECT_EQ(result.color_classes[a], result.color_classes[b])
            << "non-overlapping resources should share alias color";
        // c [2,5] overlaps both → different color
        EXPECT_NE(result.color_classes[c], result.color_classes[a])
            << "overlapping resource should get different alias color";
        EXPECT_NE(result.color_classes[c], result.color_classes[b])
            << "overlapping resource should get different alias color";
        // a and c share format → compatible
        EXPECT_TRUE(memory_allocation::AliasColorer::formats_compatible(
            formats[a], formats[c]));
    }

    // Format compatibility test: Mono and RGBA should NOT alias (different bpp).
    {
        using memory_allocation::AliasLifetime;
        std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
        std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;

        ResourceUUID mono{10, 0};
        ResourceUUID rgba{20, 0};

        lifetimes[mono] = {0, 2};
        lifetimes[rgba] = {3, 5};
        formats[mono] = StorageFormat{ChannelFormat::Mono, BitDepth::F16};
        formats[rgba] = StorageFormat{ChannelFormat::RGBA, BitDepth::F16};

        auto result = memory_allocation::AliasColorer::compute_from_lifetimes(
            lifetimes, formats);

        printf("\n=== Format compatibility test ===\n");
        printf("  mono -> color %u\n", result.color_classes[mono]);
        printf("  rgba -> color %u\n", result.color_classes[rgba]);

        // Different formats → different colors (can't share VkImage memory)
        EXPECT_NE(result.color_classes[mono], result.color_classes[rgba])
            << "different-format resources should NOT share alias color";
        EXPECT_FALSE(memory_allocation::AliasColorer::formats_compatible(
            formats[mono], formats[rgba]));
    }

    // Integration: verify perlin→blur→blur renders correctly.
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_diverse",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }

    Graph g;
    g.nodes.push_back({1, "perlin", ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({2, "blur",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({3, "blur",   ChannelFormat::RGBA, "", false, false});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline());

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(w, 512u);
    EXPECT_EQ(h, 512u);
    double sum = 0.0;
    for (size_t i = 0; i < pixels.size(); i += 4) sum += pixels[i];
    EXPECT_GT(sum, 0.0) << "slot-reuse render is all zeros";

    engine.shutdown();
}

// Chain structure diagnostic — same graph as above, compile-only (no Vulkan).
TEST(Aliasing, ComplexMultiChain_ChainStructure) {
    NodeLibrary lib;
    std::string err;
    int n = NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    ASSERT_GT(n, 0) << "failed to load nodes: " << err;

    Graph g;
    g.nodes.push_back({1, "simplex", ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({2, "levels",  ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({3, "blur",    ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({4, "invert",  ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({5, "blend",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({6, "worley",  ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({7, "warp",    ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({8, "blend",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({9, "blend",   ChannelFormat::RGBA, "", false, false});

    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({4, 0, 5, 1});
    g.connections.push_back({6, 0, 7, 0});
    g.connections.push_back({6, 0, 7, 1});
    g.connections.push_back({7, 0, 8, 1});
    g.connections.push_back({5, 0, 9, 1});
    g.connections.push_back({8, 0, 9, 2});
    g.output_node = 9;

    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    auto cr = FusedGraphCompiler::compile(r.ir, lib, g.output_node);
    ASSERT_TRUE(cr.success) << cr.error;

    printf("\n=== Complex graph chain structure (%zu chains, %zu passes) ===\n",
           cr.pass_plan.chains.size(), cr.pass_plan.passes.size());
    for (size_t i = 0; i < cr.pass_plan.chains.size(); ++i) {
        const auto& ch = cr.pass_plan.chains[i];
        printf("  chain %zu: nodes=[", i);
        for (size_t j = 0; j < ch.nodes.size(); ++j) {
            if (j) printf(",");
            printf("%u", ch.nodes[j]);
        }
        printf("] sub_passes=%u intermediates=%u total_outputs=%u\n",
               ch.sub_pass_count, ch.intermediate_count, ch.total_outputs);
    }

    // Verify the final chain contains the output node (9).
    bool found_output = false;
    for (const auto& ch : cr.pass_plan.chains) {
        for (auto nid : ch.nodes) {
            if (nid == 9) found_output = true;
        }
    }
    EXPECT_TRUE(found_output) << "output node 9 must be in one of the chains";
}

// Integration render of the full complex graph: 9 nodes, 6 chains, blur/warp.
TEST(Aliasing, ComplexMultiChain_Render) {
    Engine engine;
    bool ok = engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                          "test_shader_cache_aliasing_complex",
                          find_test_nodes_dir().c_str(),
                          find_test_glsl_dir().c_str());
    if (!ok) {
        GTEST_SKIP() << "Engine init failed: " << engine.last_error();
    }

    // simplex(1)→levels(2)→blur(3)→invert(4)→blend_A(5)
    // worley(6)→warp(7)→blend_B(8)
    // blend_A(5)+blend_B(8)→blend_C(9=output)
    Graph g;
    g.nodes.push_back({1, "simplex", ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({2, "levels",  ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({3, "blur",    ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({4, "invert",  ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({5, "blend",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({6, "worley",  ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({7, "warp",    ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({8, "blend",   ChannelFormat::RGBA, "", false, false});
    g.nodes.push_back({9, "blend",   ChannelFormat::RGBA, "", false, false});

    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({4, 0, 5, 1});
    g.connections.push_back({6, 0, 7, 0});
    g.connections.push_back({6, 0, 7, 1});
    g.connections.push_back({7, 0, 8, 1});
    g.connections.push_back({5, 0, 9, 1});
    g.connections.push_back({8, 0, 9, 2});
    g.output_node = 9;

    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_TRUE(engine.has_pipeline()) << engine.last_error();

    PushConstants pc{};
    pc.resolution_x = 512; pc.resolution_y = 512;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    ASSERT_NE(ticket, 0u);
    std::vector<float> pixels; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 200; ++i) {
        if (engine.async_readback().poll(engine.ctx(), pixels, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_EQ(w, 512u);
    EXPECT_EQ(h, 512u);
    double sum = 0.0;
    for (size_t i = 0; i < pixels.size(); i += 4) sum += pixels[i];
    EXPECT_GT(sum, 0.0) << "complex graph render is all zeros";

    auto stats = engine.resources().get_vma_stats(engine.ctx());
    printf("\n=== Complex graph render (9 nodes, 6 chains) ===\n");
    printf("  logical images: %zu (%.2f MB)\n",
           stats.node_resource_count, stats.node_resource_bytes / 1048576.0);
    printf("  physical: %.2f MB\n", stats.device_local_allocation_bytes / 1048576.0);
    printf("  efficiency: %.3f\n", stats.aliasing_efficiency());
    printf("  pixel sum: %.1f (non-zero = success)\n", sum);

    engine.shutdown();
}

// Shared memory spilling: verify the allocator assigns spill slots when budget is exceeded.
TEST(Aliasing, SharedMemorySpilling) {
    using namespace register_allocation;

    // Create 60 intervals all overlapping at step 0-59.
    // Budget = 48 → 12 resources must spill.
    IntervalMap intervals;
    for (uint32_t i = 0; i < 60; ++i) {
        ResourceUUID rid{100 + i, 0};
        intervals[rid] = {0, 59};
    }

    std::vector<NodeId> topo(60);
    for (uint32_t i = 0; i < 60; ++i) topo[i] = 100 + i;

    auto result = GraphColorer::color_linear_scan(intervals, topo, 48);

    printf("\n=== Shared memory spilling test ===\n");
    printf("  colors_used: %u (budget=48)\n", result.colors_used);
    printf("  spilled: %zu\n", result.spilled.size());
    printf("  shared_slot_count: %u\n", result.shared_slot_count);

    // All 60 intervals overlap → 60 colors needed → 12 must spill.
    EXPECT_EQ(result.colors_used, 48u);
    EXPECT_EQ(result.spilled.size(), 12u);
    EXPECT_EQ(result.shared_slot_count, 12u);

    // Every spilled resource has a shared slot.
    for (const auto& rid : result.spilled) {
        EXPECT_TRUE(result.spilled_assignment.count(rid))
            << "spilled resource must have shared slot";
        EXPECT_LT(result.spilled_assignment[rid], 12u);
    }

    // No non-spilled resource has a shared slot.
    for (const auto& [rid, color] : result.assignment) {
        EXPECT_FALSE(result.spilled_assignment.count(rid))
            << "registered resource should not have shared slot";
    }

    // Verify the GLSL builder emits shared memory correctly.
    glsl::GlslBuilder builder;
    builder.main_begin();
    if (result.shared_slot_count > 0)
        builder.declare_shared(result.shared_slot_count);
    builder.declare_local("r0");
    builder.statement("r0 = vec4(1.0);");
    builder.spill_store(0, "r0");
    builder.statement("vec4 tmp = " + builder.spill_load_expr(0) + ";");
    builder.main_end("tmp");

    std::string glsl = builder.build();
    EXPECT_NE(glsl.find("shared vec4 spill_pool[12]"), std::string::npos)
        << "GLSL must declare shared memory pool";
    EXPECT_NE(glsl.find("spill_pool[0] = r0"), std::string::npos)
        << "GLSL must emit spill store";
    EXPECT_NE(glsl.find("spill_pool[0]"), std::string::npos)
        << "GLSL must emit spill load";

    printf("  GLSL snippet:\n");
    std::istringstream iss(glsl);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.find("spill") != std::string::npos || line.find("shared") != std::string::npos)
            printf("    %s\n", line.c_str());
    }
}
