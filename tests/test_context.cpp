#include <gtest/gtest.h>
#include "engine/VulkanContext.hpp"
#include "engine/VulkanCheck.hpp"

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
