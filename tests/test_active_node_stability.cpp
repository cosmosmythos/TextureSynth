#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "test_assets.hpp"
#include <thread>
#include <chrono>
#include <vector>
#include <cmath>

using namespace te;

namespace {

bool init_engine(Engine& engine, const char* cache_name) {
    return engine.init(VK_NULL_HANDLE, nullptr, 0, true, cache_name,
                       find_test_nodes_dir().c_str(),
                       find_test_glsl_dir().c_str());
}

bool wait_for_pipeline(Engine& engine, int timeout_ms = 5000) {
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        engine.poll_pending_compiles();
        if (engine.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return engine.has_pipeline();
}

bool render(Engine& engine, uint64_t gen,
            std::vector<float>& px, int timeout_ms = 5000) {
    if (!wait_for_pipeline(engine)) return false;
    PushConstants pc{};
    pc.resolution_x = engine.output().extent().width;
    pc.resolution_y = engine.output().extent().height;
    pc.seed = 1; pc.time = 0.0f;
    uint64_t ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    if (ticket == 0) return false;
    uint64_t og = 0;
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        uint32_t w = 0, h = 0;
        if (engine.async_readback().poll(engine.ctx(), px, w, h, og)) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

double mean_r(const std::vector<float>& px) {
    double sum = 0;
    size_t n = px.size() / 4;
    for (size_t i = 0; i + 3 < px.size(); i += 4)
        sum += px[i];
    return n > 0 ? sum / n : 0.0;
}

bool all_zero(const std::vector<float>& px) {
    for (float v : px) if (v != 0.0f) return false;
    return true;
}

double max_pixel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (d > mx) mx = d;
    }
    return mx;
}

} // anonymous namespace

TEST(ActiveNodeStability, PerlinInvert_SwitchSourceAndOutput) {
    Engine engine;
    if (!init_engine(engine, "cache_active_stability_perlin_invert"))
        GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    // Render invert.
    std::vector<float> px_invert;
    ASSERT_TRUE(render(engine, gen, px_invert));
    ASSERT_FALSE(all_zero(px_invert)) << "invert all-black";
    double r_inv1 = mean_r(px_invert);

    // Switch to perlin.
    uint64_t gen_p = engine.set_active_node(1);
    ASSERT_NE(gen_p, 0u) << engine.last_error();
    std::vector<float> px_perlin;
    ASSERT_TRUE(render(engine, gen_p, px_perlin));
    ASSERT_FALSE(all_zero(px_perlin)) << "perlin all-black";
    EXPECT_GT(max_pixel_diff(px_invert, px_perlin), 1e-4)
        << "invert and perlin are pixel-identical";

    // Switch back to invert — must match first render.
    uint64_t gen_i2 = engine.set_active_node(2);
    ASSERT_NE(gen_i2, 0u) << engine.last_error();
    std::vector<float> px_inv2;
    ASSERT_TRUE(render(engine, gen_i2, px_inv2));
    EXPECT_NEAR(r_inv1, mean_r(px_inv2), 0.01)
        << "invert output changed after round-trip";

    engine.shutdown();
}

TEST(ActiveNodeStability, BlendTwoSource_SwitchBetweenAllThree) {
    Engine engine;
    if (!init_engine(engine, "cache_active_stability_blend3"))
        GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "simplex"});
    g.nodes.push_back({3, "blend"});
    g.connections.push_back({1, 0, 3, 1});
    g.connections.push_back({2, 0, 3, 2});
    g.output_node = 3;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();
    engine.update_node_params_by_id(3, {0.0f, 0.5f});

    // Render blend, perlin, simplex.
    std::vector<float> px_blend, px_perlin, px_simplex;
    ASSERT_TRUE(render(engine, gen, px_blend));
    ASSERT_FALSE(all_zero(px_blend)) << "blend all-black";

    ASSERT_TRUE(render(engine, engine.set_active_node(1), px_perlin));
    ASSERT_FALSE(all_zero(px_perlin)) << "perlin all-black";

    ASSERT_TRUE(render(engine, engine.set_active_node(2), px_simplex));
    ASSERT_FALSE(all_zero(px_simplex)) << "simplex all-black";

    // All three must differ.
    EXPECT_GT(max_pixel_diff(px_blend, px_perlin),  1e-4) << "blend == perlin";
    EXPECT_GT(max_pixel_diff(px_blend, px_simplex), 1e-4) << "blend == simplex";
    EXPECT_GT(max_pixel_diff(px_perlin, px_simplex), 1e-4) << "perlin == simplex";

    // Round-trip back to blend.
    std::vector<float> px_blend2;
    ASSERT_TRUE(render(engine, engine.set_active_node(3), px_blend2));
    EXPECT_NEAR(mean_r(px_blend), mean_r(px_blend2), 0.01)
        << "blend changed after round-trip";

    engine.shutdown();
}

TEST(ActiveNodeStability, PerlinInvert_FiveSwitchesAreDeterministic) {
    Engine engine;
    if (!init_engine(engine, "cache_active_stability_determinism"))
        GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 1;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    std::vector<double> perlin_r, invert_r;

    // Render initial perlin.
    std::vector<float> px;
    ASSERT_TRUE(render(engine, gen, px));
    ASSERT_FALSE(all_zero(px)) << "initial perlin all-black";
    perlin_r.push_back(mean_r(px));

    // Switch: perlin(1) → invert(2) → perlin → invert → perlin.
    uint64_t ids[] = {2, 1, 2, 1};
    bool is_invert[] = {true, false, true, false};

    for (int i = 0; i < 4; ++i) {
        uint64_t new_gen = engine.set_active_node(ids[i]);
        ASSERT_NE(new_gen, 0u);
        ASSERT_TRUE(render(engine, new_gen, px));
        ASSERT_FALSE(all_zero(px)) << "step " << i << " all-black";
        double r = mean_r(px);
        (is_invert[i] ? invert_r : perlin_r).push_back(r);
    }

    ASSERT_EQ(perlin_r.size(), 3u);
    ASSERT_EQ(invert_r.size(), 2u);
    for (int i = 1; i < 3; ++i)
        EXPECT_NEAR(perlin_r[0], perlin_r[i], 0.01) << "perlin render " << i << " differs";
    EXPECT_NEAR(invert_r[0], invert_r[1], 0.01) << "invert renders differ";

    engine.shutdown();
}

TEST(ActiveNodeStability, PerlinBlur_CrossChainSwitch) {
    Engine engine;
    if (!init_engine(engine, "cache_active_stability_perlin_blur"))
        GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "blur"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    // Render blur.
    std::vector<float> px_blur;
    ASSERT_TRUE(render(engine, gen, px_blur));
    ASSERT_FALSE(all_zero(px_blur)) << "blur all-black";
    double r_blur1 = mean_r(px_blur);

    // Switch to perlin.
    std::vector<float> px_perlin;
    ASSERT_TRUE(render(engine, engine.set_active_node(1), px_perlin));
    ASSERT_FALSE(all_zero(px_perlin)) << "perlin all-black";
    EXPECT_NE(r_blur1, mean_r(px_perlin)) << "blur and perlin are identical";

    // Round-trip back to blur.
    std::vector<float> px_blur2;
    ASSERT_TRUE(render(engine, engine.set_active_node(2), px_blur2));
    EXPECT_NEAR(r_blur1, mean_r(px_blur2), 0.01)
        << "blur changed after round-trip";

    engine.shutdown();
}

TEST(ActiveNodeStability, ComplexGraph_FourNodeSwitch) {
    Engine engine;
    if (!init_engine(engine, "cache_active_stability_complex4"))
        GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "simplex"});
    g.nodes.push_back({2, "levels"});
    g.nodes.push_back({3, "invert"});
    g.nodes.push_back({4, "blend"});
    g.nodes.push_back({5, "worley"});
    g.nodes.push_back({6, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 1});
    g.connections.push_back({5, 0, 6, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({6, 0, 4, 2});
    g.output_node = 4;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    // Render each target node.
    struct Step { uint64_t id; const char* name; };
    Step targets[] = {{4,"blend"}, {1,"simplex"}, {5,"worley"}, {3,"invert"}};
    std::vector<std::vector<float>> results;
    std::vector<double> means;

    for (auto& t : targets) {
        uint64_t tg = (t.id == 4) ? gen : engine.set_active_node(t.id);
        ASSERT_NE(tg, 0u) << "set_active_node(" << t.id << ") failed";
        std::vector<float> px;
        ASSERT_TRUE(render(engine, tg, px));
        ASSERT_FALSE(all_zero(px)) << t.name << " all-black";
        means.push_back(mean_r(px));
        results.push_back(std::move(px));
    }

    // All four must be pixel-different.
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            EXPECT_GT(max_pixel_diff(results[i], results[j]), 1e-4)
                << targets[i].name << " == " << targets[j].name;

    // Round-trip back to blend.
    std::vector<float> px_blend2;
    ASSERT_TRUE(render(engine, engine.set_active_node(4), px_blend2));
    EXPECT_NEAR(means[0], mean_r(px_blend2), 0.01) << "blend changed after round-trip";

    engine.shutdown();
}

TEST(ActiveNodeStability, PerlinInvert_RapidSwitching) {
    Engine engine;
    if (!init_engine(engine, "cache_active_stability_rapid"))
        GTEST_SKIP() << engine.last_error();

    Graph g;
    g.nodes.push_back({1, "perlin"});
    g.nodes.push_back({2, "invert"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 1;
    uint64_t gen = engine.set_graph(g);
    ASSERT_NE(gen, 0u) << engine.last_error();

    // Render reference invert.
    std::vector<float> ref_px;
    uint64_t ref_gen = engine.set_active_node(2);
    ASSERT_TRUE(render(engine, ref_gen, ref_px));
    double ref_r = mean_r(ref_px);

    // 10 rapid switches: 2→1→2→1→...→2
    for (int i = 0; i < 10; ++i) {
        engine.set_active_node((i % 2 == 0) ? 2 : 1);
        engine.poll_pending_compiles();
    }

    // Final state should be output=2 (invert).
    std::vector<float> final_px;
    ASSERT_TRUE(render(engine, engine.compile_generation(), final_px));
    EXPECT_NEAR(mean_r(final_px), ref_r, 0.05)
        << "after 10 switches, invert differs from reference";

    engine.shutdown();
}
