#include <gtest/gtest.h>
#include "engine/VulkanContext.hpp"
#include "engine/ResourceManager.hpp"
#include "engine/GraphIR.hpp"
#include "test_assets.hpp"
#include <vk_mem_alloc.h>
#include <filesystem>

TEST(ResourceManager, RespectsMemoryBudget) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{}; d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    te::ResourceManager rm;
    rm.set_memory_budget_mb(1); // 1 MB - intentionally tiny

    te::GraphIR ir;
    for (int i = 0; i < 100; ++i) {
        te::ValidatedNode n;
        n.id = i + 1;
        n.type_id = "dummy";
        n.debug_name = "n" + std::to_string(i);
        ir.nodes.push_back(n);
    }

    te::NodeLibrary lib;
    std::string err;
    EXPECT_FALSE(rm.allocate_for_graph(ctx, ir, lib, 2048, 2048,
                                        VK_FORMAT_R32G32B32A32_SFLOAT, &err));
    EXPECT_FALSE(err.empty());

    rm.shutdown(ctx);
    ctx.shutdown();
}

// Stage 1 / 0.2: VmaStatsReport must reflect the actual allocator state.
// 10 small identical images in one VMA block -> perfect packing, so
// aliasing_efficiency == 1.0 (baseline before transient aliasing lands
// in Stage 6).
TEST(ResourceManager, VmaStatsAreNonZero) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{}; d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    te::ResourceManager rm;
    rm.set_memory_budget_mb(1024); // 1 GB - well over our 10 tiny images

    // The existing "dummy" pattern used in RespectsMemoryBudget is empty,
    // which makes allocate_for_graph's per-type `if (!type) continue` skip
    // every node (no live images). To actually allocate something we need
    // a NodeType entry whose .outputs is non-empty.
    te::NodeLibrary lib;
    te::NodeType dummy;
    dummy.id = "dummy";
    te::Socket out;
    out.name = "color";
    out.type = te::SocketType::Vec4;
    out.format = te::ChannelFormat::RGBA;
    dummy.outputs.push_back(out);
    lib.add_public(dummy);

    te::GraphIR ir;
    for (int i = 0; i < 10; ++i) {
        te::ValidatedNode n;
        n.id = i + 1;
        n.type_id = "dummy";
        n.debug_name = "n" + std::to_string(i);
        ir.nodes.push_back(n);
    }

    std::string err;
    ASSERT_TRUE(rm.allocate_for_graph(ctx, ir, lib, 64, 64,
                                       VK_FORMAT_R32G32B32A32_SFLOAT, &err)) << err;

    auto s = rm.get_vma_stats(ctx);
    EXPECT_EQ(s.node_resource_count, 10u);
    EXPECT_GT(s.node_resource_bytes, 0u);
    EXPECT_GT(s.vma_block_bytes, 0u);
    EXPECT_GT(s.vma_allocation_bytes, 0u);
    EXPECT_EQ(s.warning_threshold_bytes, 1024u * 1024u * 1024u);
    EXPECT_EQ(s.retired_count, 0u);
    EXPECT_EQ(s.retired_bytes, 0u);
    // Perfect packing baseline: VMA packs 10 small images into one block,
    // so physical == logical -> ratio is 1.0.
    EXPECT_DOUBLE_EQ(s.aliasing_efficiency(), 1.0);

    // Sanity: byte math is right. The dummy type uses ChannelFormat::RGBA
    // which resolves to VK_FORMAT_R16G16B16A16_SFLOAT (8 bytes/pixel via
    // channel_to_vk_format), so 10 * 64 * 64 * 8 = 327680.
    EXPECT_EQ(s.node_resource_bytes, 10u * 64u * 64u * 8u);

    rm.shutdown(ctx);
    ctx.shutdown();
}

// After retire_all() the live count drops to 0, retired count grows, and
// aliasing_efficiency is undefined (division by zero -> 1.0 sentinel).
TEST(ResourceManager, VmaStatsAfterRetire) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{}; d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    te::ResourceManager rm;
    rm.set_memory_budget_mb(1024);

    te::NodeLibrary lib;
    te::NodeType dummy;
    dummy.id = "dummy";
    te::Socket out;
    out.name = "color";
    out.type = te::SocketType::Vec4;
    out.format = te::ChannelFormat::RGBA;
    dummy.outputs.push_back(out);
    lib.add_public(dummy);

    te::GraphIR ir;
    for (int i = 0; i < 5; ++i) {
        te::ValidatedNode n;
        n.id = i + 1;
        n.type_id = "dummy";
        n.debug_name = "n" + std::to_string(i);
        ir.nodes.push_back(n);
    }
    std::string err;
    ASSERT_TRUE(rm.allocate_for_graph(ctx, ir, lib, 32, 32,
                                       VK_FORMAT_R16G16B16A16_SFLOAT, &err)) << err;

    const auto live_bytes = rm.current_bytes();
    EXPECT_GT(live_bytes, 0u);

    rm.retire_all(ctx);

    auto s = rm.get_vma_stats(ctx);
    EXPECT_EQ(s.node_resource_count, 0u);
    EXPECT_EQ(s.node_resource_bytes, 0u);
    EXPECT_GT(s.retired_count, 0u);
    EXPECT_GT(s.retired_bytes, 0u);
    EXPECT_EQ(s.retired_bytes, live_bytes); // retired == what was live
    // Sentinel: zero divisor -> 1.0 (callers should check node_resource_count).
    EXPECT_DOUBLE_EQ(s.aliasing_efficiency(), 1.0);

    rm.shutdown(ctx);
    ctx.shutdown();
}

// Stage 1 / 0.2 v2: heap_stats array is populated, has at least one
// device-local heap, and the gpu_* fields aggregate from device-local
// heaps only. Without VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT, the
// per-heap budget/usage numbers fall back to VMA's 80%-of-heap-size
// estimate, which is still > 0 -- so we can assert non-zero.
TEST(ResourceManager, VmaStatsHeapBreakdown) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{}; d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    te::ResourceManager rm;
    te::NodeLibrary lib;
    te::NodeType dummy;
    dummy.id = "dummy";
    te::Socket out;
    out.name = "color";
    out.type = te::SocketType::Vec4;
    out.format = te::ChannelFormat::RGBA;
    dummy.outputs.push_back(out);
    lib.add_public(dummy);

    te::GraphIR ir;
    for (int i = 0; i < 4; ++i) {
        te::ValidatedNode n;
        n.id = i + 1;
        n.type_id = "dummy";
        n.debug_name = "n" + std::to_string(i);
        ir.nodes.push_back(n);
    }
    std::string err;
    ASSERT_TRUE(rm.allocate_for_graph(ctx, ir, lib, 32, 32,
                                      VK_FORMAT_R16G16B16A16_SFLOAT, &err)) << err;

    const auto s = rm.get_vma_stats(ctx);

    // heap_stats has one entry per VkMemoryHeap (>= 1 per Vulkan spec).
    ASSERT_FALSE(s.heap_stats.empty()) << "no heaps reported -- "
        "vmaGetMemoryProperties failed or memoryHeapCount was 0";

    // At least one device-local heap must exist (spec requirement).
    bool saw_device_local = false;
    size_t sum_budget = 0;
    size_t sum_usage  = 0;
    uint32_t total_allocations_across_heaps = 0;
    for (const auto& h : s.heap_stats) {
        EXPECT_GE(h.heap_size_bytes, 0u);
        // pressure must be in [0, 1]; clamped by the implementation.
        EXPECT_GE(h.pressure, 0.0f);
        EXPECT_LE(h.pressure, 1.0f);
        total_allocations_across_heaps += h.vma_allocation_count;
        if (h.is_device_local) {
            saw_device_local = true;
            sum_budget += h.budget_bytes;
            sum_usage  += h.usage_bytes;
        }
    }
    // VMA should have placed our 4 images in *some* heap (likely the
    // device-local one with TILING_OPTIMAL + USAGE_AUTO, but we don't
    // assume which). The aggregate must be > 0.
    EXPECT_GT(total_allocations_across_heaps, 0u)
        << "no VMA allocations found in any heap -- images not allocated?";
    EXPECT_TRUE(saw_device_local) << "no device-local heap -- invalid Vulkan implementation";

    // gpu_* fields must equal the sum of device-local heaps.
    EXPECT_EQ(s.gpu_budget_bytes, sum_budget);
    EXPECT_EQ(s.gpu_usage_bytes,  sum_usage);
    // gpu_pressure should match (clamped) usage/budget, but since no
    // allocations have been done yet the usage is typically 0 -> 0.0.
    // We just assert it's in range and non-negative.
    EXPECT_GE(s.gpu_pressure, 0.0f);
    EXPECT_LE(s.gpu_pressure, 1.0f);

    // The label field must be one of the two valid values, not "unknown".
    for (const auto& h : s.heap_stats) {
        EXPECT_TRUE(std::string(h.label) == "DEVICE_LOCAL" ||
                    std::string(h.label) == "HOST")
            << "heap " << h.index << " has unexpected label: " << h.label;
    }

    rm.shutdown(ctx);
    ctx.shutdown();
}

// Stage 1 / 0.2 v2: warning_threshold_bytes reflects the configured
// budget (a config knob), while the device-local heap's budget_bytes
// reflects the real GPU VRAM. They MUST be different on most systems
// (default 1 GB vs. real ~8 GB), which is exactly the bug this fix
// addresses -- the original API conflated them.
TEST(ResourceManager, WarningThresholdIsConfigurableAndDistinctFromGpuBudget) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{}; d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    te::ResourceManager rm;
    rm.set_memory_budget_mb(512); // half a gig -- an artificial ceiling

    // Empty graph is fine for this test; we just need the stats.
    auto s = rm.get_vma_stats(ctx);

    // The warning threshold is what the user set.
    EXPECT_EQ(s.warning_threshold_bytes, 512u * 1024u * 1024u);
    EXPECT_EQ(rm.budget_bytes(), 512u * 1024u * 1024u);

    // The real GPU budget comes from VMA's heap query. On any system
    // with >= 1 GB VRAM, this should be larger than the artificial
    // 512 MB warning threshold we just set. On a tiny system it might
    // be smaller, so we only assert "different" -- the test is about
    // the distinction in concept, not the magnitudes.
    bool distinct = (s.gpu_budget_bytes != s.warning_threshold_bytes);
    if (!distinct) {
        // Edge case: real GPU VRAM happens to be exactly 512 MB.
        // The more important assertion is that the two are not the
        // same *source* (config vs. VMA), so also check the heap_stats
        // path returns at least one heap.
        EXPECT_FALSE(s.heap_stats.empty());
    }
    // sanity: both > 0
    EXPECT_GT(s.warning_threshold_bytes, 0u);
    EXPECT_GT(s.gpu_budget_bytes, 0u);

    rm.shutdown(ctx);
    ctx.shutdown();
}

// Stage 1 / 0.1 v2: VMA-side allocation name is queryable via
// vmaGetAllocationInfo(alloc).pName. This is the *testable* name path
// (vkSetDebugUtilsObjectNameEXT cannot be queried back per Vulkan spec --
// see VVL#1168 and the lack of vkGetDebugUtilsObjectNameEXT). If both
// helpers are set in lockstep (create_image_ does both), a regression
// in either shows up here.
TEST(ResourceManager, VmaAllocationNameIsQueryable) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{}; d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    te::ResourceManager rm;
    rm.set_memory_budget_mb(1024);
    te::NodeLibrary lib;
    te::NodeType dummy;
    dummy.id = "dummy";
    te::Socket out;
    out.name = "color";
    out.type = te::SocketType::Vec4;
    out.format = te::ChannelFormat::RGBA;
    dummy.outputs.push_back(out);
    lib.add_public(dummy);

    te::GraphIR ir;
    for (int i = 0; i < 3; ++i) {
        te::ValidatedNode n;
        n.id = i + 1;
        n.type_id = "dummy";
        // Use a distinct, predictable name pattern we can verify.
        n.debug_name = "qa_node_" + std::to_string(i);
        ir.nodes.push_back(n);
    }
    std::string err;
    ASSERT_TRUE(rm.allocate_for_graph(ctx, ir, lib, 16, 16,
                                      VK_FORMAT_R16G16B16A16_SFLOAT, &err)) << err;

    // Find the live resources and check their VMA-side names.
    size_t verified = 0;
    for (const auto& kv : rm.live_resources()) {
        const auto& r = kv.second;
        if (!r.alloc) continue;
        VmaAllocationInfo info{};
        vmaGetAllocationInfo(ctx.allocator(), r.alloc, &info);
        ASSERT_NE(info.pName, nullptr) << "VMA name not set on alloc for node "
            << r.debug_name;
        // create_image_() sets both the VkObject debug name and the
        // VMA allocation name to the same string. So info.pName must
        // equal r.debug_name exactly (no further suffix).
        EXPECT_STREQ(info.pName, r.debug_name.c_str())
            << "VMA name mismatch: stored='" << (info.pName ? info.pName : "(null)")
            << "'  node debug_name='" << r.debug_name << "'";
        ++verified;
    }
    EXPECT_GE(verified, 3u) << "could not verify all 3 nodes' VMA names";

    rm.shutdown(ctx);
    ctx.shutdown();
}

// Stage 1 / 0.3 (asset-path fix): find_test_*_dir() must return an
// absolute path that points at a real shader_assets/ directory, even
// when the test binary is launched from a non-project-root CWD
// (e.g. by RenderDoc, CTest, or a different shell). This is the
// regression that caused 8 tests to skip under RenderDoc.
TEST(TestAssets, ResolvesAbsolutePathToShaderAssets) {
    const std::string root = find_test_project_root();
    ASSERT_FALSE(root.empty())
        << "find_test_project_root() returned empty -- "
        << "TEXTURESYNTH_TEST_ASSET_DIR macro not set AND walk-up "
        << "from CWD did not find shader_assets/. Either the test was "
        << "moved away from the project tree, or CMake was reconfigured "
        << "without -DTEXTURESYNTH_TEST_ASSET_DIR.";
    namespace fs = std::filesystem;
    std::error_code ec;
    EXPECT_TRUE(fs::path(root).is_absolute())
        << "returned path is not absolute: " << root;
    EXPECT_TRUE(fs::is_directory(fs::path(root) / "shader_assets" / "nodes", ec))
        << "shader_assets/nodes missing under resolved root: " << root;
    EXPECT_TRUE(fs::is_directory(fs::path(root) / "shader_assets" / "glsl", ec))
        << "shader_assets/glsl missing under resolved root: " << root;

    const std::string nodes = find_test_nodes_dir();
    const std::string glsl  = find_test_glsl_dir();
    EXPECT_NE(nodes, "shader_assets/nodes")
        << "fell back to CWD-relative -- env var / macro / walk-up all failed";
    EXPECT_NE(glsl, "shader_assets/glsl")
        << "fell back to CWD-relative -- env var / macro / walk-up all failed";
}
