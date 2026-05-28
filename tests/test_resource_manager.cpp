#include <gtest/gtest.h>
#include "engine/VulkanContext.hpp"
#include "engine/ResourceManager.hpp"
#include "engine/GraphIR.hpp"

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

    std::string err;
    EXPECT_FALSE(rm.allocate_for_graph(ctx, ir, 2048, 2048,
                                        VK_FORMAT_R32G32B32A32_SFLOAT, &err));
    EXPECT_FALSE(err.empty());

    rm.shutdown(ctx);
    ctx.shutdown();
}
