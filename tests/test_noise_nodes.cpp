#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "test_assets.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>

using namespace te;

// ============================================================================
// Noise node smoke tests â€” validates the PCG3D-based noise_common revamp.
//
// Tiling is tested via time-shift: render at time=0 and time=T where T*speed
// equals exactly one period. The two images must be pixel-identical.
//
// All tests share ONE engine instance to avoid VkInstance churn / GPU OOM.
// ============================================================================

class NoiseNodes : public ::testing::Test {
protected:
    static Engine engine_;
    static bool engine_ready_;
    static uint64_t current_gen_;

    static void SetUpTestSuite() {
        engine_ready_ = engine_.init(VK_NULL_HANDLE, nullptr, 0, true,
                                     "test_noise_nodes",
                                     find_test_nodes_dir().c_str(),
                                     find_test_glsl_dir().c_str());
    }

    static void TearDownTestSuite() {
        if (engine_ready_) engine_.shutdown();
    }

    uint64_t submit(Graph& g) {
        current_gen_ = engine_.set_graph(g);
        return current_gen_;
    }

    bool wait_pipeline(int ms = 3000) {
        for (int i = 0; i * 10 < ms; ++i) {
            engine_.poll_pending_compiles();
            if (engine_.has_pipeline()) return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return engine_.has_pipeline();
    }

    bool readback(std::vector<float>& px, uint32_t& w, uint32_t& h,
                  float time = 0.0f, int ms = 3000) {
        PushConstants pc{};
        pc.resolution_x = 256; pc.resolution_y = 256;
        pc.seed = 1; pc.time = time;
        uint64_t ticket = engine_.async_readback().submit(
            engine_.ctx(), engine_, pc, current_gen_);
        if (ticket == 0) return false;
        uint64_t og = 0;
        for (int i = 0; i * 10 < ms; ++i) {
            if (engine_.async_readback().poll(engine_.ctx(), px, w, h, og))
                return true;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        return false;
    }
};

Engine NoiseNodes::engine_;
bool NoiseNodes::engine_ready_ = false;
uint64_t NoiseNodes::current_gen_ = 0;

namespace {

bool has_nan_inf(const std::vector<float>& px) {
    for (float v : px)
        if (std::isnan(v) || std::isinf(v)) return true;
    return false;
}

double avg_brightness(const std::vector<float>& px) {
    if (px.empty()) return 0.0;
    double sum = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4)
        sum += (px[i] + px[i+1] + px[i+2]) / 3.0;
    return sum / (px.size() / 4);
}

// Tiling test: two images rendered with time-shift of exactly one period
// must be pixel-identical. Uses speed=1 and time=period.
bool check_tiling_via_time_shift(Engine& engine, uint64_t gen,
                                  float period, uint32_t res = 256) {
    // Read at time=0
    PushConstants pc0{};
    pc0.resolution_x = res; pc0.resolution_y = res;
    pc0.seed = 1; pc0.time = 0.0f;
    auto t0 = engine.async_readback().submit(engine.ctx(), engine, pc0, gen);
    if (t0 == 0) return false;
    std::vector<float> px0; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 300; ++i) {
        if (engine.async_readback().poll(engine.ctx(), px0, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (px0.empty()) return false;

    // Read at time=period (shifts noise by exactly one period)
    PushConstants pc1{};
    pc1.resolution_x = res; pc1.resolution_y = res;
    pc1.seed = 1; pc1.time = period;
    auto t1 = engine.async_readback().submit(engine.ctx(), engine, pc1, gen);
    if (t1 == 0) return false;
    std::vector<float> px1;
    for (int i = 0; i < 300; ++i) {
        if (engine.async_readback().poll(engine.ctx(), px1, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (px1.empty()) return false;

    if (px0.size() != px1.size()) return false;

    // Allow tiny float epsilon from interpolation.
    for (size_t i = 0; i < px0.size(); ++i) {
        if (std::abs(px0[i] - px1[i]) > 0.001f) return false;
    }
    return true;
}

} // anon

// ===== Value =====

TEST_F(NoiseNodes, Value_ProducesValidOutput) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "value"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h)) << "readback timed out";
    ASSERT_EQ(px.size(), w * h * 4u);
    EXPECT_FALSE(has_nan_inf(px)) << "NaN/Inf";
    EXPECT_GT(avg_brightness(px), 0.0) << "all-black";
    for (float v : px) { EXPECT_GE(v, -0.01f); EXPECT_LE(v, 1.01f); }
}

TEST_F(NoiseNodes, Value_Tiles) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "value"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 8.0f;
    engine_.update_node_params_by_name(1,
        {{"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
         {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period))
        << "value noise does not tile (time-shift mismatch)";
}

// ===== Perlin =====

TEST_F(NoiseNodes, Perlin_ProducesValidOutput) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "perlin"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h));
    EXPECT_FALSE(has_nan_inf(px)) << "NaN/Inf";
    EXPECT_GT(avg_brightness(px), 0.0) << "all-black";
    for (float v : px) { EXPECT_GE(v, -0.01f); EXPECT_LE(v, 1.01f); }
}

TEST_F(NoiseNodes, Perlin_Tiles) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "perlin"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 8.0f;
    engine_.update_node_params_by_name(1,
        {{"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
         {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period))
        << "perlin noise does not tile";
}

// ===== Simplex =====

TEST_F(NoiseNodes, Simplex_ProducesValidOutput) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "simplex"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h));
    EXPECT_FALSE(has_nan_inf(px)) << "NaN/Inf";
    EXPECT_GT(avg_brightness(px), 0.0) << "all-black";
    for (float v : px) { EXPECT_GE(v, -0.01f); EXPECT_LE(v, 1.01f); }
}

TEST_F(NoiseNodes, Simplex_Tiles) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "simplex"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 8.0f;
    engine_.update_node_params_by_name(1,
        {{"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
         {"roughness", 0.5f}, {"speed", 1.0f}, {"rotation", 0.0f},
         {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period))
        << "simplex noise does not tile";
}

// ===== Worley =====

TEST_F(NoiseNodes, Worley_ProducesValidOutput) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h));
    EXPECT_FALSE(has_nan_inf(px)) << "NaN/Inf";
    EXPECT_GT(avg_brightness(px), 0.0) << "all-black";
    for (float v : px) { EXPECT_GE(v, -0.01f); EXPECT_LE(v, 1.01f); }
}

TEST_F(NoiseNodes, Worley_Tiles) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 8.0f;
    engine_.update_node_params_by_name(1,
        {{"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
         {"roughness", 0.5f}, {"jitter", 0.5f}, {"speed", 1.0f},
         {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period))
        << "worley noise does not tile";
}

// ===== Gabor =====

TEST_F(NoiseNodes, Gabor_ProducesValidOutput) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "gabor"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h));
    EXPECT_FALSE(has_nan_inf(px)) << "NaN/Inf";
    EXPECT_GT(avg_brightness(px), 0.0) << "all-black";
    for (float v : px) { EXPECT_GE(v, -0.01f); EXPECT_LE(v, 1.01f); }
}

TEST_F(NoiseNodes, Gabor_Tiles) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "gabor"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 16.0f;
    engine_.update_node_params_by_name(1,
        {{"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
         {"roughness", 0.5f}, {"frequency", 4.0f}, {"bandwidth", 4.0f},
         {"anisotropy", 0.0f}, {"angle", 0.0f}, {"speed", 1.0f},
         {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period))
        << "gabor noise does not tile";
}

// ===== White Noise =====

TEST_F(NoiseNodes, WhiteNoise_ProducesValidOutput) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "white"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h));
    EXPECT_FALSE(has_nan_inf(px)) << "NaN/Inf";
    EXPECT_GT(avg_brightness(px), 0.0) << "all-black";
    for (float v : px) { EXPECT_GE(v, -0.01f); EXPECT_LE(v, 1.01f); }
}

TEST_F(NoiseNodes, WhiteNoise_Tiles) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "white"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 8.0f;
    engine_.update_node_params_by_name(1,
        {{"resolution", period}, {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period))
        << "white noise does not tile";
}

