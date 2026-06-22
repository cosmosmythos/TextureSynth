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
// Noise node smoke tests — validates the PCG3D-based noise_common revamp.
// Each noise type renders a single frame; we assert no NaN/Inf, non-trivial
// output, and values within [0,1]. Tiling correctness lives in noise_common.glsl
// (uses GLSL-spec mod()); it is not re-verified here.
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
