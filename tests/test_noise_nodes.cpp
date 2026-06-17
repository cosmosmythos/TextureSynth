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

TEST_F(NoiseNodes, Value_SeedSensitivity) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "value"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    engine_.update_node_params_by_name(1, {{"seed", 1.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::vector<float> px1; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px1, w, h));
    engine_.update_node_params_by_name(1, {{"seed", 999.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::vector<float> px2; uint32_t w2 = 0, h2 = 0;
    ASSERT_TRUE(readback(px2, w2, h2));
    ASSERT_EQ(px1.size(), px2.size());
    double diff = 0;
    for (size_t i = 0; i < px1.size(); ++i) diff += std::abs(px1[i] - px2[i]);
    EXPECT_GT(diff, 0.0) << "different seeds produce identical output";
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

TEST_F(NoiseNodes, Worley_JitterZero) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 4.0f;
    engine_.update_node_params_by_name(1,
        {{"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
         {"roughness", 0.5f}, {"jitter", 0.0f}, {"speed", 1.0f},
         {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h));
    EXPECT_FALSE(has_nan_inf(px));
    EXPECT_GT(avg_brightness(px), 0.0);
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period));
}

TEST_F(NoiseNodes, Worley_JitterFull) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    const float period = 8.0f;
    engine_.update_node_params_by_name(1,
        {{"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
         {"roughness", 0.5f}, {"jitter", 1.0f}, {"speed", 1.0f},
         {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    std::vector<float> px; uint32_t w = 0, h = 0;
    ASSERT_TRUE(readback(px, w, h));
    EXPECT_FALSE(has_nan_inf(px));
    EXPECT_GT(avg_brightness(px), 0.0);
    EXPECT_TRUE(check_tiling_via_time_shift(engine_, current_gen_, period));
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

// Edge-matching diagnostic: renders noise and checks that the first and last
// columns are pixel-identical. With correct uv mapping (coord / (res-1)),
// pixel(255,y) maps to uv=1.0 → p=period → same as pixel(0,y) at uv=0.
bool check_edge_match(const std::vector<float>& px, uint32_t w, uint32_t h,
                       float eps = 0.001f, bool verbose = false) {
    if (w < 2 || h == 0 || px.size() != w * h * 4) return false;
    float max_diff = 0.0f;
    uint32_t worst_y = 0, worst_c = 0;
    float worst_l = 0, worst_r = 0;
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t c = 0; c < 4; ++c) {
            float left  = px[(y * w + 0) * 4 + c];
            float right = px[(y * w + (w - 1)) * 4 + c];
            float diff = std::abs(left - right);
            if (diff > max_diff) {
                max_diff = diff;
                worst_y = y; worst_c = c;
                worst_l = left; worst_r = right;
            }
            if (diff > eps) {
                if (verbose && y < 5)
                    printf("  y=%u c=%u: left=%.6f right=%.6f diff=%.6f\n", y, c, left, right, diff);
                return false;
            }
        }
    }
    return true;
}

// Diagnostic version: prints edge diff stats, always returns true
void diagnose_edge_match(const std::vector<float>& px, uint32_t w, uint32_t h) {
    if (w < 2 || h == 0 || px.size() != w * h * 4) return;
    float max_diff = 0.0f, sum_diff = 0.0f;
    uint32_t worst_y = 0, worst_c = 0, count = 0;
    float worst_l = 0, worst_r = 0;
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t c = 0; c < 4; ++c) {
            float left  = px[(y * w + 0) * 4 + c];
            float right = px[(y * w + (w - 1)) * 4 + c];
            float diff = std::abs(left - right);
            sum_diff += diff;
            count++;
            if (diff > max_diff) {
                max_diff = diff;
                worst_y = y; worst_c = c;
                worst_l = left; worst_r = right;
            }
        }
    }
    float avg_diff = sum_diff / (float)count;
    printf("  EDGE DIAGNOSTIC: avg_diff=%.8f max_diff=%.8f "
           "(worst: y=%u c=%u left=%.6f right=%.6f)\n",
           avg_diff, max_diff, worst_y, worst_c, worst_l, worst_r);
    // Print first 5 rows detail
    for (uint32_t y = 0; y < std::min(h, 5u); ++y) {
        printf("  row y=%u:", y);
        for (uint32_t c = 0; c < 4; ++c) {
            float left  = px[(y * w + 0) * 4 + c];
            float right = px[(y * w + (w - 1)) * 4 + c];
            float diff = std::abs(left - right);
            printf(" [%u] L=%.6f R=%.6f diff=%.8f", c, left, right, diff);
        }
        printf("\n");
    }
}

// Render a specific noise type at a given period and check edge-match.
bool edge_match_noise(Engine& engine, uint64_t gen,
                      const char* type, float period,
                      std::unordered_map<std::string, float> extra_params,
                      uint32_t res = 256) {
    engine.set_resolution(res, res);
    // Build param dict
    std::unordered_map<std::string, float> params = {
        {"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
        {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f}
    };
    for (auto& [k, v] : extra_params) params[k] = v;

    engine.update_node_params_by_name(1, params);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine.poll_pending_compiles();

    PushConstants pc{};
    pc.resolution_x = res; pc.resolution_y = res;
    pc.seed = 1; pc.time = 0.0f;
    auto ticket = engine.async_readback().submit(engine.ctx(), engine, pc, gen);
    if (ticket == 0) return false;
    std::vector<float> px; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 300; ++i) {
        if (engine.async_readback().poll(engine.ctx(), px, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (px.empty() || w != res || h != res) return false;
    return check_edge_match(px, w, h);
}

// ===== Edge-matching tests for non-power-of-2 periods =====

// Raw diagnostic: compare pixel (0,0) and pixel (255,0) for ALL noise types.
// Prints exact float bit representations to detect precision issues.
TEST_F(NoiseNodes, EdgeDiagnostic_AllTypes) {
    ASSERT_TRUE(engine_ready_);
    engine_.set_resolution(256, 256);

    struct TestCase { const char* type; std::unordered_map<std::string, float> extra; };
    TestCase cases[] = {
        {"value", {}},
        {"perlin", {}},
        {"simplex", {{"rotation", 0.0f}}},
        {"worley", {{"jitter", 0.5f}}},
        {"white", {}},
    };
    for (auto& tc : cases) {
        Graph g; g.nodes.push_back({1, tc.type}); g.output_node = 1;
        ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());

        engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
            {"period", 3.0f}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
            {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f},
        });
        // Merge extra params
        for (auto& [k, v] : tc.extra)
            engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{{k, v}});

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        engine_.poll_pending_compiles();

        PushConstants pc{};
        pc.resolution_x = 256; pc.resolution_y = 256;
        pc.seed = 1; pc.time = 0.0f;
        auto ticket = engine_.async_readback().submit(
            engine_.ctx(), engine_, pc, current_gen_);
        ASSERT_NE(ticket, 0u);
        std::vector<float> px; uint32_t w = 0, h = 0; uint64_t og = 0;
        for (int i = 0; i < 300; ++i) {
            if (engine_.async_readback().poll(engine_.ctx(), px, w, h, og)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_FALSE(px.empty()) << tc.type;
        ASSERT_EQ(w, 256u); ASSERT_EQ(h, 256u);

        printf("[%s]:\n", tc.type);
        diagnose_edge_match(px, w, h);
    }
}

TEST_F(NoiseNodes, Worley_EdgeDiagnostic_Period3) {
    ASSERT_TRUE(engine_ready_);
    engine_.set_resolution(256, 256);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
        {"period", 3.0f}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
        {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f},
        {"jitter", 0.5f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    PushConstants pc{};
    pc.resolution_x = 256; pc.resolution_y = 256;
    pc.seed = 1; pc.time = 0.0f;
    auto ticket = engine_.async_readback().submit(
        engine_.ctx(), engine_, pc, current_gen_);
    ASSERT_NE(ticket, 0u);
    std::vector<float> px; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 300; ++i) {
        if (engine_.async_readback().poll(engine_.ctx(), px, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_FALSE(px.empty());
    diagnose_edge_match(px, w, h);
    // Period=3 should be perfectly tiling at the pixel level
    EXPECT_TRUE(check_edge_match(px, w, h))
        << "worley edge mismatch at period=3 (see diagnostic above)";
}

TEST_F(NoiseNodes, Worley_EdgeDiagnostic_Period8) {
    ASSERT_TRUE(engine_ready_);
    engine_.set_resolution(256, 256);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
        {"period", 8.0f}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
        {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f},
        {"jitter", 0.5f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();
    PushConstants pc{};
    pc.resolution_x = 256; pc.resolution_y = 256;
    pc.seed = 1; pc.time = 0.0f;
    auto ticket = engine_.async_readback().submit(
        engine_.ctx(), engine_, pc, current_gen_);
    ASSERT_NE(ticket, 0u);
    std::vector<float> px; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 300; ++i) {
        if (engine_.async_readback().poll(engine_.ctx(), px, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_FALSE(px.empty());
    diagnose_edge_match(px, w, h);
    EXPECT_TRUE(check_edge_match(px, w, h))
        << "worley edge mismatch at period=8";
}

TEST_F(NoiseNodes, Worley_EdgeMatch_Period3) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    EXPECT_TRUE(edge_match_noise(engine_, current_gen_, "worley", 3.0f,
        std::unordered_map<std::string, float>{{"jitter", 0.5f}}))
        << "worley edge mismatch at period=3";
}

TEST_F(NoiseNodes, Worley_EdgeMatch_Period5) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    EXPECT_TRUE(edge_match_noise(engine_, current_gen_, "worley", 5.0f,
        std::unordered_map<std::string, float>{{"jitter", 0.5f}}))
        << "worley edge mismatch at period=5";
}

TEST_F(NoiseNodes, Worley_EdgeMatch_Period7) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    EXPECT_TRUE(edge_match_noise(engine_, current_gen_, "worley", 7.0f,
        std::unordered_map<std::string, float>{{"jitter", 0.5f}}))
        << "worley edge mismatch at period=7";
}

TEST_F(NoiseNodes, Worley_EdgeMatch_Period9) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    EXPECT_TRUE(edge_match_noise(engine_, current_gen_, "worley", 9.0f,
        std::unordered_map<std::string, float>{{"jitter", 0.5f}}))
        << "worley edge mismatch at period=9";
}

TEST_F(NoiseNodes, Value_EdgeMatch_Period3) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "value"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    EXPECT_TRUE(edge_match_noise(engine_, current_gen_, "value", 3.0f, {}))
        << "value edge mismatch at period=3";
}

TEST_F(NoiseNodes, Perlin_EdgeMatch_Period3) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "perlin"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    EXPECT_TRUE(edge_match_noise(engine_, current_gen_, "perlin", 3.0f, {}))
        << "perlin edge mismatch at period=3";
}

TEST_F(NoiseNodes, Simplex_EdgeMatch_Period4) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "simplex"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    EXPECT_TRUE(edge_match_noise(engine_, current_gen_, "simplex", 4.0f,
        std::unordered_map<std::string, float>{{"rotation", 0.0f}}))
        << "simplex edge mismatch at period=4";
}

TEST_F(NoiseNodes, WhiteNoise_EdgeMatch_Period3) {
    ASSERT_TRUE(engine_ready_);
    Graph g; g.nodes.push_back({1, "white"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    // White noise uses "resolution" param, not "period"
    engine_.update_node_params_by_name(1,
        {{"resolution", 3.0f}, {"seed", 0.0f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    engine_.poll_pending_compiles();

    PushConstants pc{};
    pc.resolution_x = 256; pc.resolution_y = 256;
    pc.seed = 1; pc.time = 0.0f;
    auto ticket = engine_.async_readback().submit(
        engine_.ctx(), engine_, pc, current_gen_);
    ASSERT_NE(ticket, 0u);
    std::vector<float> px; uint32_t w = 0, h = 0; uint64_t og = 0;
    for (int i = 0; i < 300; ++i) {
        if (engine_.async_readback().poll(engine_.ctx(), px, w, h, og)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ASSERT_FALSE(px.empty());
    EXPECT_TRUE(check_edge_match(px, w, h))
        << "white noise edge mismatch at resolution=3";
}

// ===== Cross-type =====

TEST_F(NoiseNodes, DifferentTypesProduceDifferentOutput) {
    ASSERT_TRUE(engine_ready_);
    auto render = [&](const char* type) {
        Graph g; g.nodes.push_back({1, type}); g.output_node = 1;
        submit(g); wait_pipeline();
        std::vector<float> px; uint32_t w = 0, h = 0;
        readback(px, w, h);
        return px;
    };
    auto v = render("value"), p = render("perlin"), wr = render("worley");
    ASSERT_EQ(v.size(), p.size());
    ASSERT_EQ(v.size(), wr.size());
    double vp = 0, vw = 0, pw = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        vp += std::abs(v[i] - p[i]);
        vw += std::abs(v[i] - wr[i]);
        pw += std::abs(p[i] - wr[i]);
    }
    EXPECT_GT(vp, 0.0) << "value == perlin";
    EXPECT_GT(vw, 0.0) << "value == worley";
    EXPECT_GT(pw, 0.0) << "perlin == worley";
}

// Row scan: render Worley at period=3 and period=4,
// print pixels 240-255 of row 0 to see transition at right edge.
TEST_F(NoiseNodes, EdgeDiagnostic_WorleyRowScan) {
    ASSERT_TRUE(engine_ready_);
    engine_.set_resolution(256, 256);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());

    float periods[] = {2.0f, 3.0f, 4.0f, 8.0f};
    for (float period : periods) {
        engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
            {"period", period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
            {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f},
            {"jitter", 0.5f}});
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        engine_.poll_pending_compiles();

        PushConstants pc{};
        pc.resolution_x = 256; pc.resolution_y = 256;
        pc.seed = 1; pc.time = 0.0f;
        auto ticket = engine_.async_readback().submit(
            engine_.ctx(), engine_, pc, current_gen_);
        ASSERT_NE(ticket, 0u);
        std::vector<float> px; uint32_t w = 0, h = 0; uint64_t og = 0;
        for (int retry = 0; retry < 300; ++retry) {
            if (engine_.async_readback().poll(engine_.ctx(), px, w, h, og)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_FALSE(px.empty());

        printf("period=%.0f row_0 col_240..255:\n", period);
        printf("  x:   ");
        for (uint32_t x = 240; x < 256; ++x) printf(" %4u", x);
        printf("\n");
        for (uint32_t c = 0; c < 3; ++c) {
            printf("  ch%u:", c);
            for (uint32_t x = 240; x < 256; ++x) {
                printf(" %5.3f", px[(0 * w + x) * 4 + c]);
            }
            printf("\n");
        }
        printf("  col0=[%.6f,%.6f,%.6f] col255=[%.6f,%.6f,%.6f]\n\n",
            px[0], px[1], px[2],
            px[(0*w+255)*4], px[(0*w+255)*4+1], px[(0*w+255)*4+2]);
    }
}

// Render SAME pixel twice to check determinism.
TEST_F(NoiseNodes, EdgeDiagnostic_WorleyDeterminism) {
    ASSERT_TRUE(engine_ready_);
    engine_.set_resolution(256, 256);
    Graph g; g.nodes.push_back({1, "worley"}); g.output_node = 1;
    ASSERT_TRUE(submit(g)); ASSERT_TRUE(wait_pipeline());
    engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
        {"period", 3.0f}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
        {"roughness", 0.5f}, {"speed", 1.0f}, {"seed", 0.0f},
        {"jitter", 0.5f}});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    engine_.poll_pending_compiles();
    PushConstants pc{};
    pc.resolution_x = 256; pc.resolution_y = 256;
    pc.seed = 1; pc.time = 0.0f;
    std::vector<float> px1, px2;
    uint32_t w = 0, h = 0;
    for (int pass = 0; pass < 2; ++pass) {
        auto ticket = engine_.async_readback().submit(
            engine_.ctx(), engine_, pc, current_gen_);
        ASSERT_NE(ticket, 0u);
        std::vector<float> px; uint64_t og = 0;
        for (int retry = 0; retry < 300; ++retry) {
            if (engine_.async_readback().poll(engine_.ctx(), px, w, h, og)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_FALSE(px.empty());
        (pass == 0 ? px1 : px2) = std::move(px);
    }
    float max_diff = 0;
    for (size_t i = 0; i < px1.size(); ++i)
        max_diff = std::max(max_diff, std::abs(px1[i] - px2[i]));
    printf("Worley determinism (pass1 vs pass2): max_diff=%.10f\n", max_diff);
    for (uint32_t c = 0; c < 3; ++c) {
        float l = px1[(0*w+0)*4+c], r = px1[(0*w+255)*4+c];
        printf("  ch%u edge: L=%.8f R=%.8f diff=%.8f\n", c, l, r, std::abs(l-r));
    }
}

// CPU simulation of the EXACT GLSL ts_pcg3d + ts_worley_tile logic.
// Reproduces the GPU computation to check if the edge mismatch is in the MATH.
static uint32_t pcg3d_x(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t v[3] = {x, y, z};
    v[0] = v[0] * 1664525u + 1013904223u;
    v[1] = v[1] * 1664525u + 1013904223u;
    v[2] = v[2] * 1664525u + 1013904223u;
    v[0] += v[1] * v[2]; v[1] += v[2] * v[0]; v[2] += v[0] * v[1];
    v[0] ^= v[0] >> 16u; v[1] ^= v[1] >> 16u; v[2] ^= v[2] >> 16u;
    v[0] += v[1] * v[2]; v[1] += v[2] * v[0]; v[2] += v[0] * v[1];
    return v[0];
}
static uint32_t pcg3d_y(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t v[3] = {x, y, z};
    v[0] = v[0] * 1664525u + 1013904223u;
    v[1] = v[1] * 1664525u + 1013904223u;
    v[2] = v[2] * 1664525u + 1013904223u;
    v[0] += v[1] * v[2]; v[1] += v[2] * v[0]; v[2] += v[0] * v[1];
    v[0] ^= v[0] >> 16u; v[1] ^= v[1] >> 16u; v[2] ^= v[2] >> 16u;
    v[0] += v[1] * v[2]; v[1] += v[2] * v[0]; v[2] += v[0] * v[1];
    return v[1];
}
static int ts_wrap_cpu(int v, int per) { return ((v % per) + per) % per; }
static float ts_hash2_cpu(int cx, int cy, uint32_t seed) {
    return float(pcg3d_x((uint32_t)cx, (uint32_t)cy, seed) & 0xFFFFu) * (1.0f / 65536.0f);
}
static float ts_hash2_vec2_x_cpu(int cx, int cy, uint32_t seed) {
    return float(pcg3d_x((uint32_t)cx, (uint32_t)cy, seed) & 0xFFFFu) * (1.0f / 65536.0f);
}
static float ts_hash2_vec2_y_cpu(int cx, int cy, uint32_t seed) {
    return float(pcg3d_y((uint32_t)cx, (uint32_t)cy, seed) & 0xFFFFu) * (1.0f / 65536.0f);
}
static float worley_cpu(float px, float py, int per, float jitter, uint32_t seed) {
    int pi_x = (int)std::floor(px), pi_y = (int)std::floor(py);
    float pf_x = px - pi_x, pf_y = py - pi_y;
    float f1 = 8.0f;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int nx = ts_wrap_cpu(pi_x + i, per);
            int ny = ts_wrap_cpu(pi_y + j, per);
            float fx = ts_hash2_vec2_x_cpu(nx, ny, seed);
            float fy = ts_hash2_vec2_y_cpu(nx, ny, seed);
            fx = 0.5f + (fx - 0.5f) * jitter;
            fy = 0.5f + (fy - 0.5f) * jitter;
            float dx = (float)i + fx - pf_x;
            float dy = (float)j + fy - pf_y;
            float d2 = dx*dx + dy*dy;
            if (d2 < f1) f1 = d2;
        }
    }
    return std::clamp(std::sqrt(f1) * 0.70710678f, 0.0f, 1.0f);
}

TEST_F(NoiseNodes, EdgeDiagnostic_CPUSimulation) {
    int per = 3;
    float jitter = 0.5f;
    uint32_t seeds[] = {1, 380, 758};
    for (uint32_t seed : seeds) {
        float v0 = worley_cpu(0.0f, 0.0f, per, jitter, seed);
        float v2 = worley_cpu(3.0f, 0.0f, per, jitter, seed);
        printf("CPU sim seed=%u: px0=%.10f px2=%.10f diff=%+.10f %s\n",
            seed, v0, v2, v0-v2,
            (std::abs(v0-v2) < 1e-6f ? "MATCH" : "MISMATCH"));
    }
    // Also test with p=2.999999 vs p=3.0 to check floor precision
    float v_exact = worley_cpu(3.0f, 0.0f, per, jitter, 1);
    float v_close  = worley_cpu(2.999999f, 0.0f, per, jitter, 1);
    printf("floor test: p=3.0 -> %.10f, p=2.999999 -> %.10f diff=%+.10f\n",
        v_exact, v_close, v_exact - v_close);

    // Verify CPU pcg3d matches GLSL ts_pcg3d for specific inputs
    uint32_t h0 = pcg3d_x(0, 0, 1);
    uint32_t h1 = pcg3d_y(0, 0, 1);
    printf("pcg3d(0,0,1): x=%u y=%u x&0xFFFF=%u y&0xFFFF=%u\n",
        h0, h1, h0 & 0xFFFFu, h1 & 0xFFFFu);
    // Compute feature point for cell (0,0) seed=1
    float fx = float(h0 & 0xFFFFu) * (1.0f / 65536.0f);
    float fy = float(h1 & 0xFFFFu) * (1.0f / 65536.0f);
    fx = 0.5f + (fx - 0.5f) * 0.5f;
    fy = 0.5f + (fy - 0.5f) * 0.5f;
    printf("cell(0,0) seed=1 feature=(%.8f, %.8f)\n", fx, fy);
    // Check: for seed=758
    uint32_t h0b = pcg3d_x(0, 0, 758);
    uint32_t h1b = pcg3d_y(0, 0, 758);
    float fxb = float(h0b & 0xFFFFu) * (1.0f / 65536.0f);
    float fyb = float(h1b & 0xFFFFu) * (1.0f / 65536.0f);
    fxb = 0.5f + (fxb - 0.5f) * 0.5f;
    fyb = 0.5f + (fyb - 0.5f) * 0.5f;
    printf("cell(0,0) seed=758 feature=(%.8f, %.8f)\n", fxb, fyb);
}
