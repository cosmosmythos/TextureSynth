#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/Engine.hpp"
#include "test_assets.hpp"
#include <cstdio>
#include <algorithm>

static void dump(const char* label, const te::Engine::BakedImage& img) {
    std::printf("=== %s [w=%u h=%u pixels=%zu] ===\n",
                label, img.width, img.height, img.pixels.size());
    if (img.pixels.empty()) return;
    uint32_t w = std::min<uint32_t>(16, img.width);
    uint32_t h = std::min<uint32_t>(4, img.height);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            size_t i = (size_t(y) * img.width + x) * 4;
            std::printf("(%2u,%u)=[%6.3f,%6.3f,%6.3f,%6.3f]  ", x, y,
                img.pixels[i+0], img.pixels[i+1], img.pixels[i+2], img.pixels[i+3]);
        }
        std::printf("\n");
    }
    double s = 0; float mn = 1e30f, mx = -1e30f;
    for (float v : img.pixels) { s += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    std::printf("sum=%.1f  min=%.3f  max=%.3f  mean=%.3f\n\n",
                s, mn, mx, s / img.pixels.size());
}

TEST(Inspector, PerlinVsInvertPerlin) {
    te::Engine engine;
    if (!engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                     "inspect_perlin_invert",
                     find_test_nodes_dir().c_str(),
                     find_test_glsl_dir().c_str())) {
        GTEST_SKIP() << "init failed";
    }
    te::Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;
    ASSERT_NE(engine.set_graph(g), 0u);
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    dump("active=invert (perlin->invert)", engine.readback_sync());

    engine.set_active_node(1);
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    dump("active=perlin (direct)", engine.readback_sync());

    engine.shutdown();
}

TEST(Inspector, SimplexChannelRange) {
    te::Engine engine;
    if (!engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                     "inspect_simplex_range",
                     find_test_nodes_dir().c_str(),
                     find_test_glsl_dir().c_str())) {
        GTEST_SKIP() << "init failed";
    }
    te::Graph g;
    g.nodes.push_back({1, "simplex"});
    g.output_node = 1;
    ASSERT_NE(engine.set_graph(g), 0u);
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    dump("simplex (direct)", engine.readback_sync());
    engine.shutdown();
}

TEST(Inspector, SimplexVsInvertSimplex) {
    te::Engine engine;
    if (!engine.init(VK_NULL_HANDLE, nullptr, 0, true,
                     "inspect_simplex_invert",
                     find_test_nodes_dir().c_str(),
                     find_test_glsl_dir().c_str())) {
        GTEST_SKIP() << "init failed";
    }
    te::Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;
    ASSERT_NE(engine.set_graph(g), 0u);
    for (int i = 0; i < 200 && !engine.has_pipeline(); ++i) {
        engine.poll_pending_compiles();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    dump("active=invert (simplex->invert)", engine.readback_sync());
    engine.shutdown();
}
