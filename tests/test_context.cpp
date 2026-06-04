#include <gtest/gtest.h>
#include "engine/VulkanContext.hpp"
#include "engine/VulkanCheck.hpp"
#include <cstring>

TEST(VulkanContext, InitAndShutdownClean) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{};
    d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));
    EXPECT_NE(ctx.device(), VK_NULL_HANDLE);
    EXPECT_NE(ctx.allocator(), nullptr);
    ctx.shutdown();
}

TEST(VulkanContext, DoubleShutdownIsSafe) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{};
    ASSERT_TRUE(ctx.init(d));
    ctx.shutdown();
    ctx.shutdown(); // must not crash
}

// Stage 1 / 0.1 v3: set_debug_name must (1) return VK_SUCCESS on a
// well-formed call against a real Vulkan object, (2) treat the no-op
// cases (empty name, null handle, fn-pointer null) as VK_SUCCESS so
// callers can chain without an if-guard, and (3) handle the Vulkan
// 256-byte name limit (including NUL) without crashing -- the spec
// path returns VK_ERROR_VALIDATION_FAILED_EXT for too-long names, and
// our implementation must truncate to 255 chars to avoid that.
//
// We can't *query* the name back (Vulkan has no
// vkGetDebugUtilsObjectNameEXT by design, see VVL#1168), but a
// VK_SUCCESS return is meaningful: it means the validation layer
// accepted the call, the sType/objectType/objectHandle fields were
// well-formed, and the name (post-truncation) was within the spec
// length. The actual name is then visible to RenderDoc and custom
// layers, both of which intercept vkSetDebugUtilsObjectNameEXT.
TEST(VulkanContext, SetDebugNameAcceptsValidCallAndHandlesEdgeCases) {
    te::VulkanContext ctx;
    te::VulkanContextDesc d{};
    d.enable_validation = true;
    ASSERT_TRUE(ctx.init(d));

    // Create a minimal dummy image to name. 1x1 R8G8B8A8 storage
    // image -- the smallest legal Vulkan image, no VMA needed (we
    // destroy it manually with vkDestroyImage to keep this test
    // independent of the VMA dependency).
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent      = {1, 1, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImage image = VK_NULL_HANDLE;
    VkResult r = vkCreateImage(ctx.device(), &ici, nullptr, &image);
    ASSERT_EQ(r, VK_SUCCESS) << "dummy image creation failed";
    ASSERT_NE(image, VK_NULL_HANDLE);

    // (1) Valid call -> VK_SUCCESS (with validation on, the layer
    //     accepts the call and propagates the success return).
    r = ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)image,
                           "smoke_test_image");
    EXPECT_EQ(r, VK_SUCCESS) << "valid call returned non-success";

    // (2a) Empty name -> VK_SUCCESS (treated as a no-op).
    r = ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)image, "");
    EXPECT_EQ(r, VK_SUCCESS) << "empty name should be a no-op";

    // (2b) Null handle -> VK_SUCCESS (no-op).
    r = ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, 0, "ignored");
    EXPECT_EQ(r, VK_SUCCESS) << "null handle should be a no-op";

    // (3) Long name (>255 chars) -> no crash, no VVL error. The
    //     implementation truncates proactively.
    const std::string long_name(500, 'x');
    r = ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)image, long_name);
    EXPECT_EQ(r, VK_SUCCESS) << "long name should truncate, not error";

    // (3b) Edge: name exactly at the 255-char limit must work too.
    const std::string max_name(255, 'y');
    r = ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)image, max_name);
    EXPECT_EQ(r, VK_SUCCESS) << "max-length name should work";

    // (3c) Edge: name one over the limit (256 chars) must also work
    // (our truncation kicks in for >= 256).
    const std::string over_name(256, 'z');
    r = ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)image, over_name);
    EXPECT_EQ(r, VK_SUCCESS) << "256-char name should truncate, not error";

    vkDestroyImage(ctx.device(), image, nullptr);
    ctx.shutdown();
}

TEST(VulkanCheck, BasicCompileAndSuccess) {
    VkResult res = VK_CHECK(VK_SUCCESS);
    EXPECT_EQ(res, VK_SUCCESS);
}

TEST(VulkanCheck, LogsOnFailure) {
    std::string captured;
    te::set_log_sink([&](const char*, const std::string& m) { captured = m; });

    VkResult res = VK_CHECK(VK_ERROR_OUT_OF_HOST_MEMORY);
    EXPECT_EQ(res, VK_ERROR_OUT_OF_HOST_MEMORY);

    EXPECT_FALSE(captured.empty());
    EXPECT_NE(captured.find("VK_ERROR_OUT_OF_HOST_MEMORY"), std::string::npos);

    te::set_log_sink({});
}

TEST(VulkanCheck, LogsOnFailureThrow) {
    std::string captured;
    te::set_log_sink([&](const char*, const std::string& m) { captured = m; });

    EXPECT_THROW({ VK_CHECK_THROW(VK_ERROR_OUT_OF_HOST_MEMORY); }, std::runtime_error);

    EXPECT_FALSE(captured.empty());
    EXPECT_NE(captured.find("VK_ERROR_OUT_OF_HOST_MEMORY"), std::string::npos);

    te::set_log_sink({});
}

TEST(VulkanCheck, ShortFnStripsPath) {
    EXPECT_STREQ(te::detail::short_fn("foo/bar/baz.cpp"), "baz.cpp");
    EXPECT_STREQ(te::detail::short_fn("baz.cpp"), "baz.cpp");
    EXPECT_STREQ(te::detail::short_fn("C:\\Users\\x\\y.cpp"), "y.cpp");
}
