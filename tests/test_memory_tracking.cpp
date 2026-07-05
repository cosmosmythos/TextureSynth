#include <gtest/gtest.h>
#include <memory>
#include "engine/Engine.hpp"
#include "engine/PushConstants.hpp"
#include "engine/ResourceManager.hpp"
#include "test_assets.hpp"
#include <chrono>
#include <thread>

namespace {

struct EngineHolder {
    std::unique_ptr<te::Engine> engine;
    bool ready = false;

    EngineHolder(const char* cache = "test_shader_cache_memtrack") {
        engine = std::make_unique<te::Engine>();
        bool ok = engine->init(VK_NULL_HANDLE, nullptr, 0,
                               /*validation*/ true, cache,
                               find_test_nodes_dir().c_str(),
                               find_test_glsl_dir().c_str());
        if (!ok) {
            engine.reset();
            return;
        }
        ready = true;
    }

    te::Engine* operator->() { return engine.get(); }
    te::Engine& operator*() { return *engine; }
    explicit operator bool() const { return ready; }
};

void wait_compile(te::Engine& engine, int max_ms = 3000) {
    for (int i = 0; i < max_ms / 10 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void pump_ticks(te::Engine& engine, int count) {
    for (int i = 0; i < count; ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

} // namespace

TEST(MemoryTracking, SingleNodeAllocations) {
    EngineHolder h("test_shader_cache_memtrack_1");
    if (!h) GTEST_SKIP() << "Engine init failed";

    te::Graph g;
    g.nodes.push_back({1, "value"});
    g.output_node = 1;

    uint64_t gen = h->set_graph(g);
    if (gen == 0) GTEST_SKIP() << "set_graph failed: " << h->last_error();

    wait_compile(*h);

    size_t live = h->resources_live_count();
    EXPECT_EQ(live, 1u) << "Single node graph: 1 live resource image";

    h->shutdown();
}

TEST(MemoryTracking, FiveNodeChainAllocations) {
    EngineHolder h("test_shader_cache_memtrack_5");
    if (!h) GTEST_SKIP() << "Engine init failed";

    te::Graph g;
    g.nodes.push_back({1, "value"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "invert"});
    g.nodes.push_back({4, "invert"});
    g.nodes.push_back({5, "invert"});
    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({2, 0, 3, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({4, 0, 5, 1});
    g.output_node = 5;

    uint64_t gen = h->set_graph(g);
    if (gen == 0) GTEST_SKIP() << "set_graph failed: " << h->last_error();

    wait_compile(*h);

    size_t live = h->resources_live_count();
    EXPECT_EQ(live, 1u) << "5-node chain: 1 live resource image (graph output only)";

    h->shutdown();
}

TEST(MemoryTracking, SixNodeChainAllocations) {
    EngineHolder h("test_shader_cache_memtrack_6");
    if (!h) GTEST_SKIP() << "Engine init failed";

    te::Graph g;
    g.nodes.push_back({1, "value"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "invert"});
    g.nodes.push_back({4, "invert"});
    g.nodes.push_back({5, "invert"});
    g.nodes.push_back({6, "invert"});
    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({2, 0, 3, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({4, 0, 5, 1});
    g.connections.push_back({5, 0, 6, 1});
    g.output_node = 6;

    uint64_t gen = h->set_graph(g);
    if (gen == 0) GTEST_SKIP() << "set_graph failed: " << h->last_error();

    wait_compile(*h);

    size_t live = h->resources_live_count();
    EXPECT_EQ(live, 1u) << "6-node chain: 1 live resource image (graph output only)";

    std::cerr << "[mem] 6-node: live=" << live
              << " live_bytes=" << h->resources().current_bytes()
              << " retired=" << h->resources().retired_count()
              << " retired_bytes=" << h->resources().retired_bytes() << std::endl;

    h->shutdown();
}

TEST(MemoryTracking, ResourceImageVsGroupOutputCount) {
    EngineHolder h("test_shader_cache_memtrack_resvgrp");
    if (!h) GTEST_SKIP() << "Engine init failed";

    te::Graph g;
    g.nodes.push_back({1, "value"});
    g.nodes.push_back({2, "invert"});
    g.nodes.push_back({3, "invert"});
    g.nodes.push_back({4, "invert"});
    g.nodes.push_back({5, "invert"});
    g.connections.push_back({1, 0, 2, 1});
    g.connections.push_back({2, 0, 3, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({4, 0, 5, 1});
    g.output_node = 5;

    uint64_t gen = h->set_graph(g);
    if (gen == 0) GTEST_SKIP() << "set_graph failed: " << h->last_error();

    wait_compile(*h);

    size_t live = h->resources_live_count();
    std::cerr << "[mem] res_vs_grp: resource_images=" << live
              << " (expect 1 — only graph output gets resource image)" << std::endl;
    EXPECT_EQ(live, 1u) << "Only graph output node gets a resource image";

    h->shutdown();
}

TEST(MemoryTracking, RetiredImagesAccumulate) {
    EngineHolder h("test_shader_cache_memtrack_retacc");
    if (!h) GTEST_SKIP() << "Engine init failed";

    te::Graph g;
    g.nodes.push_back({1, "value"});
    g.output_node = 1;

    uint64_t gen1 = h->set_graph(g);
    if (gen1 == 0) GTEST_SKIP() << "set_graph failed: " << h->last_error();
    wait_compile(*h);

    size_t live1 = h->resources_live_count();
    size_t retired1 = h->resources().retired_count();
    std::cerr << "[mem] after 1st set_graph: live=" << live1
              << " retired=" << retired1 << std::endl;

    g.nodes[0].id = 2;
    g.output_node = 2;
    h->set_graph(g);
    wait_compile(*h);

    size_t live2 = h->resources_live_count();
    size_t retired2 = h->resources().retired_count();
    size_t retbytes2 = h->resources().retired_bytes();
    std::cerr << "[mem] after 2nd set_graph: live=" << live2
              << " retired=" << retired2
              << " retired_bytes=" << retbytes2 << std::endl;

    EXPECT_EQ(live2, 1u) << "Should have 1 live image after re-submit";
    EXPECT_GE(retired2, 1u) << "Should have at least 1 retired image";

    h->shutdown();
}

TEST(MemoryTracking, RetiredImagesEventuallyDestroyed) {
    EngineHolder h("test_shader_cache_memtrack_retdel");
    if (!h) GTEST_SKIP() << "Engine init failed";

    te::Graph g;
    g.nodes.push_back({1, "value"});
    g.output_node = 1;

    uint64_t gen1 = h->set_graph(g);
    if (gen1 == 0) GTEST_SKIP() << "set_graph failed: " << h->last_error();
    wait_compile(*h);

    g.nodes[0].id = 2;
    g.output_node = 2;
    h->set_graph(g);
    wait_compile(*h);

    size_t retired_before = h->resources().retired_count();
    std::cerr << "[mem] retired before pumping: " << retired_before << std::endl;
    EXPECT_GE(retired_before, 1u);

    pump_ticks(*h, 10);

    size_t retired_after = h->resources().retired_count();
    std::cerr << "[mem] retired after 10 ticks: " << retired_after << std::endl;
    EXPECT_EQ(retired_after, 0u) << "All retired destroyed after RETIRE_FRAMES (4) ticks";

    h->shutdown();
}
