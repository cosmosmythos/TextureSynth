#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "test_assets.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdio>

using namespace te;

// ============================================================================
// SIMPLEX NOISE DEBUGGING SUITE
//
// Goal: 10000% understand WHY simplex only tiles at even periods.
//
// Strategy:
//   1. CPU simulation of the EXACT GLSL ts_simplex_tile algorithm
//   2. Test edge match, gradient continuity, cell structure at every period
//   3. Compare "buggy" (even-Y rounding) vs "fixed" (no rounding) paths
//   4. GPU tests to confirm CPU findings match GPU output
// ============================================================================

namespace {

// ===== EXACT CPU REPLICATION OF GLSL FUNCTIONS =====

static const float TS_HASH_TO_ANGLE = 21.548f;
static const float TS_SIMPLEX_NORM  = 10.9f;
static const float TS_PI_OVER_PHI   = 1.9098593171f;

struct Vec3 { float x, y, z; };
struct uvec3 { uint32_t x, y, z; };

static uvec3 pcg3d(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t v[3] = {x, y, z};
    v[0] = v[0] * 1664525u + 1013904223u;
    v[1] = v[1] * 1664525u + 1013904223u;
    v[2] = v[2] * 1664525u + 1013904223u;
    v[0] += v[1] * v[2]; v[1] += v[2] * v[0]; v[2] += v[0] * v[1];
    v[0] ^= v[0] >> 16u; v[1] ^= v[1] >> 16u; v[2] ^= v[2] >> 16u;
    v[0] += v[1] * v[2]; v[1] += v[2] * v[0]; v[2] += v[0] * v[1];
    return {v[0], v[1], v[2]};
}

static float wrap_f(float v, float period) {
    float r = std::fmod(v, period);
    if (r < 0.0f) r += period;
    return r;
}

// Exact CPU copy of ts_simplex_tile from noise_common.glsl:310-373
struct SimplexResult {
    float value;
    float gx, gy;
};

static SimplexResult ts_simplex_tile_cpu(
    float x, float y, int period_x, int period_y,
    float alpha, uint32_t seed)
{
    // Skew to simplex coordinate space
    float uv_x = x + y * 0.5f;
    float uv_y = y;

    float i0x = std::floor(uv_x);
    float i0y = std::floor(uv_y);
    float f0x = uv_x - i0x;
    float f0y = uv_y - i0y;

    float cmp = (f0y <= f0x) ? 1.0f : 0.0f;
    float o1x = cmp;
    float o1y = 1.0f - cmp;

    float i1x = i0x + o1x;
    float i1y = i0y + o1y;
    float i2x = i0x + 1.0f;
    float i2y = i0y + 1.0f;

    // Back-transform corners to texture space
    float v0x = i0x - i0y * 0.5f;
    float v0y = i0y;
    float v1x = v0x + o1x - o1y * 0.5f;
    float v1y = v0y + o1y;
    float v2x = v0x + 0.5f;
    float v2y = v0y + 1.0f;

    // Displacement vectors
    float x0x = x - v0x, x0y = y - v0y;
    float x1x = x - v1x, x1y = y - v1y;
    float x2x = x - v2x, x2y = y - v2y;

    // Wrap corners to period (EXACT GLSL mod)
    float xw[3] = {v0x, v1x, v2x};
    float yw[3] = {v0y, v1y, v2y};
    xw[0] = wrap_f(xw[0], (float)period_x);
    xw[1] = wrap_f(xw[1], (float)period_x);
    xw[2] = wrap_f(xw[2], (float)period_x);
    yw[0] = wrap_f(yw[0], (float)period_y);
    yw[1] = wrap_f(yw[1], (float)period_y);
    yw[2] = wrap_f(yw[2], (float)period_y);

    // Hash lookup (EXACT GLSL floor(xw + 0.5*yw + 0.5), floor(yw + 0.5))
    float iuf[3], ivf[3];
    for (int i = 0; i < 3; i++) {
        iuf[i] = std::floor(xw[i] + 0.5f * yw[i] + 0.5f);
        ivf[i] = std::floor(yw[i] + 0.5f);
    }

    // pcg3d hash for each corner
    auto hash_corner = [&](int idx) -> float {
        uint32_t iu = (uint32_t)(int)iuf[idx];
        uint32_t iv = (uint32_t)(int)ivf[idx];
        uvec3 h = pcg3d(iu, iv, seed);
        return (float)(h.x & 0xFFFFu) * (1.0f / 65536.0f);
    };

    float h01[3];
    h01[0] = hash_corner(0);
    h01[1] = hash_corner(1);
    h01[2] = hash_corner(2);

    // Gradient angles
    float psi[3];
    for (int i = 0; i < 3; i++)
        psi[i] = h01[i] * TS_HASH_TO_ANGLE + alpha;

    float gx[3], gy[3];
    for (int i = 0; i < 3; i++) {
        gx[i] = std::cos(psi[i]);
        gy[i] = std::sin(psi[i]);
    }

    // Radial decay kernel
    float dot0 = x0x*x0x + x0y*x0y;
    float dot1 = x1x*x1x + x1y*x1y;
    float dot2 = x2x*x2x + x2y*x2y;
    float w[3] = {
        std::max(0.8f - dot0, 0.0f),
        std::max(0.8f - dot1, 0.0f),
        std::max(0.8f - dot2, 0.0f)
    };
    float w2[3], w4[3];
    for (int i = 0; i < 3; i++) {
        w2[i] = w[i] * w[i];
        w4[i] = w2[i] * w2[i];
    }

    float gdotx[3] = {
        gx[0]*x0x + gy[0]*x0y,
        gx[1]*x1x + gy[1]*x1y,
        gx[2]*x2x + gy[2]*x2y
    };
    float n = w4[0]*gdotx[0] + w4[1]*gdotx[1] + w4[2]*gdotx[2];

    // Analytical gradient
    float w3[3];
    for (int i = 0; i < 3; i++) w3[i] = w2[i] * w[i];
    float dw[3];
    for (int i = 0; i < 3; i++) dw[i] = -8.0f * w3[i] * gdotx[i];

    float dn0x = w4[0]*gx[0] + dw[0]*x0x;
    float dn0y = w4[0]*gy[0] + dw[0]*x0y;
    float dn1x = w4[1]*gx[1] + dw[1]*x1x;
    float dn1y = w4[1]*gy[1] + dw[1]*x1y;
    float dn2x = w4[2]*gx[2] + dw[2]*x2x;
    float dn2y = w4[2]*gy[2] + dw[2]*x2y;

    SimplexResult r;
    r.gx = TS_SIMPLEX_NORM * (dn0x + dn1x + dn2x);
    r.gy = TS_SIMPLEX_NORM * (dn0y + dn1y + dn2y);
    r.value = TS_SIMPLEX_NORM * n;
    return r;
}

// ts_fbm_simplex: the BUGGY version with even-Y rounding (current code)
static float ts_fbm_simplex_buggy(
    float px, float py, int period_x, int period_y,
    float octaves, float lacunarity, float gain,
    float alpha, uint32_t seed)
{
    float weight = 1.0f;
    float g = gain * std::min(lacunarity, 1.0f);
    float freq = 1.0f;
    int lac_int = (int)lacunarity;

    // BUG: even-Y rounding
    int per_x = std::max(period_x, 1);
    int per_y = std::max((period_y / 2) * 2, 2);

    uint32_t os = seed;
    float oct_alpha = alpha;
    auto r0 = ts_simplex_tile_cpu(px * freq, py * freq, per_x, per_y, oct_alpha, os);
    float base = r0.value;
    float norm = 1.0f;
    float oct = 1.0f;
    if (oct >= octaves) return base;

    do {
        weight *= g;
        oct += 1.0f;
        if (oct >= octaves) weight *= 1.0f - (oct - octaves);
        freq *= lacunarity;
        per_x *= lac_int;
        per_y *= lac_int;
        per_y = std::max((per_y / 2) * 2, 2);
        oct_alpha = alpha + oct * TS_PI_OVER_PHI;
        os += 379u;
        auto r = ts_simplex_tile_cpu(px * freq, py * freq, per_x, per_y, oct_alpha, os);
        base += weight * r.value;
        norm += weight;
    } while (oct < octaves);

    return base / std::max(norm, 1e-6f);
}

// ts_fbm_simplex: the FIXED version (no even-Y rounding)
static float ts_fbm_simplex_fixed(
    float px, float py, int period_x, int period_y,
    float octaves, float lacunarity, float gain,
    float alpha, uint32_t seed)
{
    float weight = 1.0f;
    float g = gain * std::min(lacunarity, 1.0f);
    float freq = 1.0f;
    int lac_int = (int)lacunarity;

    // FIXED: no even-Y rounding
    int per_x = std::max(period_x, 1);
    int per_y = std::max(period_y, 1);

    uint32_t os = seed;
    float oct_alpha = alpha;
    auto r0 = ts_simplex_tile_cpu(px * freq, py * freq, per_x, per_y, oct_alpha, os);
    float base = r0.value;
    float norm = 1.0f;
    float oct = 1.0f;
    if (oct >= octaves) return base;

    do {
        weight *= g;
        oct += 1.0f;
        if (oct >= octaves) weight *= 1.0f - (oct - octaves);
        freq *= lacunarity;
        per_x *= lac_int;
        per_y *= lac_int;
        oct_alpha = alpha + oct * TS_PI_OVER_PHI;
        os += 379u;
        auto r = ts_simplex_tile_cpu(px * freq, py * freq, per_x, per_y, oct_alpha, os);
        base += weight * r.value;
        norm += weight;
    } while (oct < octaves);

    return base / std::max(norm, 1e-6f);
}

// Compute simplex value at exact UV coordinate, matching how node_simplex.glsl works
static float eval_simplex_buggy(float uv_x, float uv_y, int period, float alpha, uint32_t seed) {
    float p_x = uv_x * (float)period;
    float p_y = uv_y * (float)period;
    return ts_fbm_simplex_buggy(p_x, p_y, period, period, 1.0f, 2.0f, 0.5f, alpha, seed);
}

static float eval_simplex_fixed(float uv_x, float uv_y, int period, float alpha, uint32_t seed) {
    float p_x = uv_x * (float)period;
    float p_y = uv_y * (float)period;
    return ts_fbm_simplex_fixed(p_x, p_y, period, period, 1.0f, 2.0f, 0.5f, alpha, seed);
}

// NEW: simulate the FULL updated pipeline (node_simplex uses ts_period_int, ts_fbm_simplex has no rounding)
static float eval_simplex_new(float uv_x, float uv_y, int period, float alpha, uint32_t seed) {
    int iper = std::max((int)std::round((float)period), 1);
    float p_x = uv_x * (float)iper;
    float p_y = uv_y * (float)iper;
    return ts_fbm_simplex_fixed(p_x, p_y, iper, iper, 1.0f, 2.0f, 0.5f, alpha, seed);
}

// Evaluate at pixel position (0-based) in a res x res image
static float eval_pixel_buggy(int px, int py, int res, int period, float alpha, uint32_t seed) {
    float uv_x = (float)px / (float)(res - 1);
    float uv_y = (float)py / (float)(res - 1);
    return eval_simplex_buggy(uv_x, uv_y, period, alpha, seed);
}

static float eval_pixel_fixed(int px, int py, int res, int period, float alpha, uint32_t seed) {
    float uv_x = (float)px / (float)(res - 1);
    float uv_y = (float)py / (float)(res - 1);
    return eval_simplex_fixed(uv_x, uv_y, period, alpha, seed);
}

// Gradient at pixel position
static void eval_pixel_grad_buggy(int px, int py, int res, int period, float alpha, uint32_t seed,
                                   float& val, float& gx, float& gy) {
    float uv_x = (float)px / (float)(res - 1);
    float uv_y = (float)py / (float)(res - 1);
    float p_x = uv_x * (float)period;
    float p_y = uv_y * (float)period;
    // Central differences for gradient
    float eps = 0.001f;
    float val_xp = ts_fbm_simplex_buggy((uv_x + eps) * period, p_y, period, period, 1.0f, 2.0f, 0.5f, alpha, seed);
    float val_xm = ts_fbm_simplex_buggy((uv_x - eps) * period, p_y, period, period, 1.0f, 2.0f, 0.5f, alpha, seed);
    float val_yp = ts_fbm_simplex_buggy(p_x, (uv_y + eps) * period, period, period, 1.0f, 2.0f, 0.5f, alpha, seed);
    float val_ym = ts_fbm_simplex_buggy(p_x, (uv_y - eps) * period, period, period, 1.0f, 2.0f, 0.5f, alpha, seed);
    val = ts_fbm_simplex_buggy(p_x, p_y, period, period, 1.0f, 2.0f, 0.5f, alpha, seed);
    gx = (val_xp - val_xm) / (2.0f * eps);
    gy = (val_yp - val_ym) / (2.0f * eps);
}

} // anon

// ============================================================================
// TEST 1: Even-Y rounding demonstration
// Shows exactly what ts_fbm_simplex does to the period for every integer 1-10
// ============================================================================
TEST(SimplexEvenYBug, PeriodRoundingTable) {
    printf("\n=== ts_fbm_simplex PERIOD ROUNDING TABLE ===\n");
    printf("%-8s | %-12s | %-12s | %-10s\n", "Input", "Bug: (X,Y)", "Fixed: (X,Y)", "Match?");
    printf("---------|-------------|-------------|----------\n");

    for (int p = 1; p <= 10; p++) {
        int bug_x = std::max(p, 1);
        int bug_y = std::max((p / 2) * 2, 2);
        int fix_x = std::max(p, 1);
        int fix_y = std::max(p, 1);
        bool match = (bug_x == fix_x && bug_y == fix_y);
        printf("%-8d | (%2d, %2d)     | (%2d, %2d)     | %s\n",
               p, bug_x, bug_y, fix_x, fix_y, match ? "YES" : "NO ***");
    }
    printf("\n");
}

// ============================================================================
// TEST 2: CPU edge match — does value at uv=0 == value at uv=1?
// This is the tiling seam. For perfect tiling, pixel(0,y) == pixel(255,y).
// ============================================================================
TEST(SimplexEvenYBug, CPUEdgeMatch_Buggy_AllPeriods) {
    printf("\n=== CPU EDGE MATCH (BUGGY) — pixel(0,y) vs pixel(255,y) ===\n");
    printf("%-8s | %-12s | %-12s | %-10s\n", "Period", "MaxDiff(R)", "MaxDiff(G)", "TILES?");
    printf("---------|-------------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period = 2; period <= 10; period++) {
        float max_diff_r = 0.0f, max_diff_g = 0.0f;
        for (int y = 0; y < RES; y++) {
            float v_left_r  = eval_pixel_buggy(0, y, RES, period, ALPHA, SEED);
            float v_right_r = eval_pixel_buggy(RES - 1, y, RES, period, ALPHA, SEED);
            float v_left_g  = eval_pixel_buggy(0, y, RES, period, ALPHA, SEED + 379u);
            float v_right_g = eval_pixel_buggy(RES - 1, y, RES, period, ALPHA, SEED + 379u);
            max_diff_r = std::max(max_diff_r, std::abs(v_left_r - v_right_r));
            max_diff_g = std::max(max_diff_g, std::abs(v_left_g - v_right_g));
        }
        bool tiles = (max_diff_r < 0.001f && max_diff_g < 0.001f);
        printf("%-8d | %-12.8f | %-12.8f | %s\n",
               period, max_diff_r, max_diff_g, tiles ? "YES" : "NO ***");
    }
}

TEST(SimplexEvenYBug, CPUEdgeMatch_Fixed_AllPeriods) {
    printf("\n=== CPU EDGE MATCH (FIXED) — pixel(0,y) vs pixel(255,y) ===\n");
    printf("%-8s | %-12s | %-12s | %-10s\n", "Period", "MaxDiff(R)", "MaxDiff(G)", "TILES?");
    printf("---------|-------------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period = 2; period <= 10; period++) {
        float max_diff_r = 0.0f, max_diff_g = 0.0f;
        for (int y = 0; y < RES; y++) {
            float v_left_r  = eval_pixel_fixed(0, y, RES, period, ALPHA, SEED);
            float v_right_r = eval_pixel_fixed(RES - 1, y, RES, period, ALPHA, SEED);
            float v_left_g  = eval_pixel_fixed(0, y, RES, period, ALPHA, SEED + 379u);
            float v_right_g = eval_pixel_fixed(RES - 1, y, RES, period, ALPHA, SEED + 379u);
            max_diff_r = std::max(max_diff_r, std::abs(v_left_r - v_right_r));
            max_diff_g = std::max(max_diff_g, std::abs(v_left_g - v_right_g));
        }
        bool tiles = (max_diff_r < 0.001f && max_diff_g < 0.001f);
        printf("%-8d | %-12.8f | %-12.8f | %s\n",
               period, max_diff_r, max_diff_g, tiles ? "YES" : "NO ***");
    }
}

// ============================================================================
// TEST 3: Top/bottom edge match (Y tiling seam)
// ============================================================================
TEST(SimplexEvenYBug, CPUVerticalEdgeMatch_Buggy) {
    printf("\n=== CPU VERTICAL EDGE (BUGGY) — pixel(x,0) vs pixel(x,255) ===\n");
    printf("%-8s | %-12s | %-10s\n", "Period", "MaxDiff", "TILES?");
    printf("---------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period = 2; period <= 10; period++) {
        float max_diff = 0.0f;
        for (int x = 0; x < RES; x++) {
            float v_top    = eval_pixel_buggy(x, 0, RES, period, ALPHA, SEED);
            float v_bottom = eval_pixel_buggy(x, RES - 1, RES, period, ALPHA, SEED);
            max_diff = std::max(max_diff, std::abs(v_top - v_bottom));
        }
        bool tiles = (max_diff < 0.001f);
        printf("%-8d | %-12.8f | %s\n", period, max_diff, tiles ? "YES" : "NO ***");
    }
}

TEST(SimplexEvenYBug, CPUVerticalEdgeMatch_Fixed) {
    printf("\n=== CPU VERTICAL EDGE (FIXED) — pixel(x,0) vs pixel(x,255) ===\n");
    printf("%-8s | %-12s | %-10s\n", "Period", "MaxDiff", "TILES?");
    printf("---------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period = 2; period <= 10; period++) {
        float max_diff = 0.0f;
        for (int x = 0; x < RES; x++) {
            float v_top    = eval_pixel_fixed(x, 0, RES, period, ALPHA, SEED);
            float v_bottom = eval_pixel_fixed(x, RES - 1, RES, period, ALPHA, SEED);
            max_diff = std::max(max_diff, std::abs(v_top - v_bottom));
        }
        bool tiles = (max_diff < 0.001f);
        printf("%-8d | %-12.8f | %s\n", period, max_diff, tiles ? "YES" : "NO ***");
    }
}

// ============================================================================
// TEST 4: Gradient continuity at seam
// For seamless tiling, the gradient direction should be continuous across
// the seam. If gradient jumps, the texture will have visible seams.
// ============================================================================
TEST(SimplexEvenYBug, GradientContinuity_Buggy) {
    printf("\n=== GRADIENT CONTINUITY AT SEAM (BUGGY) ===\n");
    printf("%-8s | %-12s | %-12s | %-12s\n", "Period", "GradJumpX", "GradJumpY", "Continuous?");
    printf("---------|-------------|-------------|-------------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period = 2; period <= 10; period++) {
        float max_gx_jump = 0.0f, max_gy_jump = 0.0f;
        for (int y = 0; y < RES; y++) {
            float val_l, gx_l, gy_l;
            float val_r, gx_r, gy_r;
            eval_pixel_grad_buggy(1, y, RES, period, ALPHA, SEED, val_l, gx_l, gy_l);
            eval_pixel_grad_buggy(RES - 2, y, RES, period, ALPHA, SEED, val_r, gx_r, gy_r);
            max_gx_jump = std::max(max_gx_jump, std::abs(gx_l - gx_r));
            max_gy_jump = std::max(max_gy_jump, std::abs(gy_l - gy_r));
        }
        bool continuous = (max_gx_jump < 0.1f && max_gy_jump < 0.1f);
        printf("%-8d | %-12.6f | %-12.6f | %s\n",
               period, max_gx_jump, max_gy_jump, continuous ? "YES" : "NO ***");
    }
}

// ============================================================================
// TEST 5: Detailed pixel comparison at seam for specific period
// Show actual pixel values side by side
// ============================================================================
TEST(SimplexEvenYBug, DetailedSeamComparison) {
    printf("\n=== DETAILED SEAM VALUES — pixel(0,y) vs pixel(255,y) ===\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period : {3, 4, 5, 6, 7, 8}) {
        printf("\n--- Period %d (BUGGY) ---\n", period);
        for (int y = 0; y < 8; y++) {
            float vl = eval_pixel_buggy(0, y, RES, period, ALPHA, SEED);
            float vr = eval_pixel_buggy(RES - 1, y, RES, period, ALPHA, SEED);
            printf("  y=%3d: px[0]=%.8f  px[255]=%.8f  diff=%+.8f\n",
                   y, vl, vr, vl - vr);
        }

        printf("--- Period %d (FIXED) ---\n", period);
        for (int y = 0; y < 8; y++) {
            float vl = eval_pixel_fixed(0, y, RES, period, ALPHA, SEED);
            float vr = eval_pixel_fixed(RES - 1, y, RES, period, ALPHA, SEED);
            printf("  y=%3d: px[0]=%.8f  px[255]=%.8f  diff=%+.8f\n",
                   y, vl, vr, vl - vr);
        }
    }
}

// ============================================================================
// TEST 6: Direct comparison — buggy vs fixed at the SAME pixel
// Shows the exact difference the even-Y rounding makes
// ============================================================================
TEST(SimplexEvenYBug, BuggyVsFixed_DirectComparison) {
    printf("\n=== BUGGY vs FIXED at same pixel ===\n");
    printf("Shows how even-Y rounding changes the output for odd periods\n\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period : {3, 4, 5, 6, 7}) {
        float max_diff = 0.0f;
        int worst_x = 0, worst_y = 0;
        for (int y = 0; y < RES; y++) {
            for (int x = 0; x < RES; x++) {
                float vb = eval_pixel_buggy(x, y, RES, period, ALPHA, SEED);
                float vf = eval_pixel_fixed(x, y, RES, period, ALPHA, SEED);
                float d = std::abs(vb - vf);
                if (d > max_diff) {
                    max_diff = d;
                    worst_x = x;
                    worst_y = y;
                }
            }
        }
        bool same = (max_diff < 0.0001f);
        printf("Period %d: max_diff=%.8f at (%d,%d) %s\n",
               period, max_diff, worst_x, worst_y,
               same ? "(identical)" : "(DIFFERENT)");
    }
}

// ============================================================================
// TEST 7: Hash lattice comparison — what does even-Y rounding do to the hash grid?
// ============================================================================
TEST(SimplexEvenYBug, HashLatticeComparison) {
    printf("\n=== HASH LATTICE: even-Y rounding changes which cells get hashed ===\n");
    printf("For period=3: bug uses Y-period=2, so Y wraps at 2, X wraps at 3\n");
    printf("This means corners at different Y positions get different hashes\n\n");

    const int period = 3;
    const int bug_y = 2; // (3/2)*2 = 2
    const int fix_y = 3;

    printf("Corner positions in texture space and their hash lookups:\n");
    printf("%-20s | %-20s | %-20s\n", "Corner", "Bug hash(iu,iv)", "Fixed hash(iu,iv)");
    printf("---------------------|---------------------|---------------------\n");

    for (int j = -1; j <= 5; j++) {
        for (int i = -1; i <= 5; i++) {
            // Compute corner position in texture space (simplified)
            float vx = (float)i - (float)j * 0.5f;
            float vy = (float)j;

            // Bug wrapping
            float bx = wrap_f(vx, (float)period);
            float by_bug = wrap_f(vy, (float)bug_y);
            float by_fix = wrap_f(vy, (float)fix_y);

            int iu_bug = (int)std::floor(bx + 0.5f * by_bug + 0.5f);
            int iv_bug = (int)std::floor(by_bug + 0.5f);
            int iu_fix = (int)std::floor(bx + 0.5f * by_fix + 0.5f);
            int iv_fix = (int)std::floor(by_fix + 0.5f);

            if (j >= 0 && j <= 3 && i >= 0 && i <= 3) {
                auto hb = pcg3d((uint32_t)iu_bug, (uint32_t)iv_bug, 1u);
                auto hf = pcg3d((uint32_t)iu_fix, (uint32_t)iv_fix, 1u);
                bool match = (hb.x == hf.x);
                printf("v=(%+.1f,%+.1f)       | iu=%2d iv=%2d h=%08x | iu=%2d iv=%2d h=%08x %s\n",
                       vx, vy,
                       iu_bug, iv_bug, hb.x,
                       iu_fix, iv_fix, hf.x,
                       match ? "" : "<<< DIFFERENT");
            }
        }
    }
}

// ============================================================================
// GPU TESTS — confirm CPU findings match GPU output
// ============================================================================

class SimplexGPUDebug : public ::testing::Test {
protected:
    static Engine engine_;
    static bool engine_ready_;
    static uint64_t current_gen_;

    static void SetUpTestSuite() {
        engine_ready_ = engine_.init(VK_NULL_HANDLE, nullptr, 0, true,
                                     "test_simplex_debug",
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

Engine SimplexGPUDebug::engine_;
bool SimplexGPUDebug::engine_ready_ = false;
uint64_t SimplexGPUDebug::current_gen_ = 0;

// ============================================================================
// GPU TEST: Edge match at every period 2-10
// ============================================================================
TEST_F(SimplexGPUDebug, EdgeMatch_AllPeriods) {
    ASSERT_TRUE(engine_ready_);
    printf("\n=== GPU EDGE MATCH — Simplex periods 2-10 ===\n");
    printf("%-8s | %-12s | %-12s | %-12s | %-10s\n",
           "Period", "MaxDiff_R", "MaxDiff_G", "MaxDiff_B", "TILES?");
    printf("---------|-------------|-------------|-------------|----------\n");

    for (int period = 2; period <= 10; period++) {
        Graph g; g.nodes.push_back({1, "simplex"}); g.output_node = 1;
        submit(g); wait_pipeline();
        engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
            {"period", (float)period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
            {"roughness", 0.5f}, {"speed", 1.0f}, {"rotation", 0.0f},
            {"seed", 0.0f}});
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

        float max_r = 0, max_g = 0, max_b = 0;
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t c = 0; c < 3; c++) {
                float left  = px[(y * w + 0) * 4 + c];
                float right = px[(y * w + (w - 1)) * 4 + c];
                float diff = std::abs(left - right);
                if (c == 0) max_r = std::max(max_r, diff);
                if (c == 1) max_g = std::max(max_g, diff);
                if (c == 2) max_b = std::max(max_b, diff);
            }
        }
        bool tiles = (max_r < 0.001f && max_g < 0.001f && max_b < 0.001f);
        printf("%-8d | %-12.8f | %-12.8f | %-12.8f | %s\n",
               period, max_r, max_g, max_b, tiles ? "YES" : "NO ***");
    }
}

// ============================================================================
// TEST: NEW PIPELINE — node snaps to even, ts_fbm_simplex has no rounding
// This is the actual code behavior after the fix.
// ============================================================================
TEST(SimplexEvenYBug, NewPipeline_VerticalEdgeMatch) {
    printf("\n=== NEW PIPELINE VERTICAL EDGE — ts_period_int + fbm_no_rounding ===\n");
    printf("%-8s | %-10s | %-12s\n", "Period", "MaxDiff", "TILES?");
    printf("---------|----------|-------------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period = 2; period <= 10; period++) {
        float max_diff = 0.0f;
        for (int x = 0; x < RES; x++) {
            float v_top    = eval_simplex_new((float)x / (RES - 1), 0.0f, period, ALPHA, SEED);
            float v_bottom = eval_simplex_new((float)x / (RES - 1), 1.0f, period, ALPHA, SEED);
            max_diff = std::max(max_diff, std::abs(v_top - v_bottom));
        }
        bool tiles = (max_diff < 0.001f);
        printf("%-8d | %-10.8f | %s\n",
               period, max_diff, tiles ? "YES" : "NO ***");
    }
}

TEST(SimplexEvenYBug, NewPipeline_HorizontalEdgeMatch) {
    printf("\n=== NEW PIPELINE HORIZONTAL EDGE — ts_period_int + fbm_no_rounding ===\n");
    printf("%-8s | %-10s | %-12s\n", "Period", "MaxDiff", "TILES?");
    printf("---------|----------|-------------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;

    for (int period = 2; period <= 10; period++) {
        float max_diff = 0.0f;
        for (int y = 0; y < RES; y++) {
            float v_left  = eval_simplex_new(0.0f, (float)y / (RES - 1), period, ALPHA, SEED);
            float v_right = eval_simplex_new(1.0f, (float)y / (RES - 1), period, ALPHA, SEED);
            max_diff = std::max(max_diff, std::abs(v_left - v_right));
        }
        bool tiles = (max_diff < 0.001f);
        printf("%-8d | %-10.8f | %s\n",
               period, max_diff, tiles ? "YES" : "NO ***");
    }
}

TEST(SimplexEvenYBug, NewPipeline_OddDiffersFromEven) {
    printf("\n=== NEW PIPELINE: period 3 vs 4, 5 vs 6 (must be DIFFERENT) ===\n");
    const int RES = 256;
    const float ALPHA = 0.0f;
    const uint32_t SEED = 1u;
    for (auto& [odd, even] : std::vector<std::pair<int,int>>{{3,4},{5,6},{7,8}}) {
        float max_diff = 0.0f;
        for (int y = 0; y < RES; y++) {
            for (int x = 0; x < RES; x++) {
                float v_odd  = eval_simplex_new((float)x / (RES - 1), (float)y / (RES - 1), odd, ALPHA, SEED);
                float v_even = eval_simplex_new((float)x / (RES - 1), (float)y / (RES - 1), even, ALPHA, SEED);
                max_diff = std::max(max_diff, std::abs(v_odd - v_even));
            }
        }
        bool different = (max_diff > 0.001f);
        printf("Period %d vs %d: max_diff=%.8f %s\n",
               odd, even, max_diff, different ? "DIFFERENT (good!)" : "IDENTICAL (bad!)");
    }
}

// ============================================================================
// GPU TEST: Vertical edge match at every period
// ============================================================================
TEST_F(SimplexGPUDebug, VerticalEdgeMatch_AllPeriods) {
    ASSERT_TRUE(engine_ready_);
    printf("\n=== GPU VERTICAL EDGE — Simplex periods 2-10 ===\n");

    for (int period = 2; period <= 10; period++) {
        Graph g; g.nodes.push_back({1, "simplex"}); g.output_node = 1;
        submit(g); wait_pipeline();
        engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
            {"period", (float)period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
            {"roughness", 0.5f}, {"speed", 1.0f}, {"rotation", 0.0f},
            {"seed", 0.0f}});
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

        float max_diff = 0;
        for (uint32_t x = 0; x < w; x++) {
            float top    = px[(0 * w + x) * 4];
            float bottom = px[((h - 1) * w + x) * 4];
            max_diff = std::max(max_diff, std::abs(top - bottom));
        }
        bool tiles = (max_diff < 0.001f);
        printf("Period %2d vertical: max_diff=%.8f %s\n", period, max_diff, tiles ? "OK" : "BROKEN");
    }
}

// ============================================================================
// GPU TEST: Time-shift tiling (the canonical test)
// ============================================================================
TEST_F(SimplexGPUDebug, TimeShiftTiling_AllPeriods) {
    ASSERT_TRUE(engine_ready_);
    printf("\n=== GPU TIME-SHIFT TILING — Simplex periods 2-10 ===\n");

    for (int period = 2; period <= 10; period++) {
        Graph g; g.nodes.push_back({1, "simplex"}); g.output_node = 1;
        submit(g); wait_pipeline();
        engine_.update_node_params_by_name(1, std::unordered_map<std::string, float>{
            {"period", (float)period}, {"octaves", 1.0f}, {"lacunarity", 2.0f},
            {"roughness", 0.5f}, {"speed", 1.0f}, {"rotation", 0.0f},
            {"seed", 0.0f}});
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        engine_.poll_pending_compiles();

        // Render at time=0
        PushConstants pc0{};
        pc0.resolution_x = 256; pc0.resolution_y = 256;
        pc0.seed = 1; pc0.time = 0.0f;
        auto t0 = engine_.async_readback().submit(engine_.ctx(), engine_, pc0, current_gen_);
        ASSERT_NE(t0, 0u);
        std::vector<float> px0; uint32_t w = 0, h = 0; uint64_t og = 0;
        for (int i = 0; i < 300; ++i) {
            if (engine_.async_readback().poll(engine_.ctx(), px0, w, h, og)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_FALSE(px0.empty());

        // Render at time=period
        PushConstants pc1{};
        pc1.resolution_x = 256; pc1.resolution_y = 256;
        pc1.seed = 1; pc1.time = (float)period;
        auto t1 = engine_.async_readback().submit(engine_.ctx(), engine_, pc1, current_gen_);
        ASSERT_NE(t1, 0u);
        std::vector<float> px1;
        for (int i = 0; i < 300; ++i) {
            if (engine_.async_readback().poll(engine_.ctx(), px1, w, h, og)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ASSERT_FALSE(px1.empty());

        float max_diff = 0;
        for (size_t i = 0; i < px0.size(); i++)
            max_diff = std::max(max_diff, std::abs(px0[i] - px1[i]));

        bool tiles = (max_diff < 0.001f);
        printf("Period %2d time-shift: max_diff=%.8f %s\n", period, max_diff, tiles ? "OK" : "BROKEN");
    }
}

// ============================================================================
// CPU TEST: Assert-based — verify the root cause
// The bug is that ts_fbm_simplex rounds period.y to even.
// For odd periods, buggy != fixed. For even periods, buggy == fixed.
// ============================================================================
TEST(SimplexEvenYBug, RootCauseAssertion) {
    printf("\n=== ROOT CAUSE ASSERTION ===\n");

    for (int period = 2; period <= 10; period++) {
        int bug_x = std::max(period, 1);
        int bug_y = std::max((period / 2) * 2, 2);
        int fix_x = std::max(period, 1);
        int fix_y = std::max(period, 1);

        if (period % 2 == 0) {
            // Even period: bug and fix produce same Y → output should be identical
            EXPECT_EQ(bug_y, fix_y) << "Even period " << period;
            float vb = eval_pixel_buggy(128, 128, 256, period, 0.0f, 1u);
            float vf = eval_pixel_fixed(128, 128, 256, period, 0.0f, 1u);
            EXPECT_NEAR(vb, vf, 1e-6f) << "Even period " << period << " output differs";
        } else {
            // Odd period: bug rounds Y down, fix keeps it → different output
            EXPECT_NE(bug_y, fix_y) << "Odd period " << period;
            float vb = eval_pixel_buggy(128, 128, 256, period, 0.0f, 1u);
            float vf = eval_pixel_fixed(128, 128, 256, period, 0.0f, 1u);
            // These SHOULD differ because the bug changes the Y period
            // (But if the pixel is far from the seam, they might still match
            //  by coincidence — so we just print, don't assert)
            printf("  Period %d (odd): buggy=%.8f fixed=%.8f diff=%+.8f %s\n",
                   period, vb, vf, vb - vf,
                   (std::abs(vb - vf) > 0.0001f) ? "CONFIRMED DIFFERENT" : "coincidentally same");
        }
    }
}

// ============================================================================
// 3D SIMPLEX GRID CPU SIMULATION
//
// Exact CPU replication of the new ts_simplex_tile in noise_common.glsl
// (3D axis-aligned simplex grid evaluated on z=0 slice).
// This is the algorithm that tiles at ANY integer period.
// ============================================================================

static const float TS_SIMPLEX_NORM_3D_CPU = 39.5f;

static SimplexResult ts_simplex_tile_3d_cpu(
    float x, float y, int period_x, int period_y, float alpha)
{
    // Promote to 3D: z=0 slice, no Z wrapping
    Vec3 x3 = {x, y, 0.0f};

    // Axis-aligned simplex grid: uvw = M * x where M = [[0,1,1],[1,0,1],[1,1,0]]
    // uvw.x = x3.y + x3.z, uvw.y = x3.x + x3.z, uvw.z = x3.x + x3.y
    Vec3 uvw = {x3.y + x3.z, x3.x + x3.z, x3.x + x3.y};

    Vec3 i0f = {std::floor(uvw.x), std::floor(uvw.y), std::floor(uvw.z)};
    Vec3 f0 = {uvw.x - i0f.x, uvw.y - i0f.y, uvw.z - i0f.z};

    // Rank-order fractional parts for tetrahedron corners
    float gx_ = (f0.x <= f0.y) ? 1.0f : 0.0f;
    float gy_ = (f0.y <= f0.z) ? 1.0f : 0.0f;
    float gz_ = (f0.x <= f0.z) ? 1.0f : 0.0f;
    float lx_ = 1.0f - gx_;
    float ly_ = 1.0f - gy_;
    float lz_ = 1.0f - gz_;

    Vec3 g = {lz_, gx_, gy_};
    Vec3 l = {lx_, ly_, gz_};

    Vec3 o1, o2;
    o1.x = std::min(g.x, l.x); o1.y = std::min(g.y, l.y); o1.z = std::min(g.z, l.z);
    o2.x = std::max(g.x, l.x); o2.y = std::max(g.y, l.y); o2.z = std::max(g.z, l.z);

    Vec3 i1f = {i0f.x + o1.x, i0f.y + o1.y, i0f.z + o1.z};
    Vec3 i2f = {i0f.x + o2.x, i0f.y + o2.y, i0f.z + o2.z};
    Vec3 i3f = {i0f.x + 1.0f, i0f.y + 1.0f, i0f.z + 1.0f};

    // Mi = [[-0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5]]
    auto Mi_mul = [](Vec3 v) -> Vec3 {
        return Vec3{
            -0.5f*v.x + 0.5f*v.y + 0.5f*v.z,
             0.5f*v.x - 0.5f*v.y + 0.5f*v.z,
             0.5f*v.x + 0.5f*v.y - 0.5f*v.z
        };
    };

    Vec3 v0 = Mi_mul(i0f);
    Vec3 v1 = Mi_mul(i1f);
    Vec3 v2 = Mi_mul(i2f);
    Vec3 v3 = Mi_mul(i3f);

    // Displacement vectors
    Vec3 x0 = {x3.x - v0.x, x3.y - v0.y, x3.z - v0.z};
    Vec3 x1 = {x3.x - v1.x, x3.y - v1.y, x3.z - v1.z};
    Vec3 x2 = {x3.x - v2.x, x3.y - v2.y, x3.z - v2.z};
    Vec3 x3_ = {x3.x - v3.x, x3.y - v3.y, x3.z - v3.z};

    // Wrap corner positions and transform back to simplex space
    auto wrap_and_hash = [&](Vec3 v) -> uvec3 {
        float vx = v.x, vy = v.y;
        if (period_x > 0) vx = wrap_f(vx, (float)period_x);
        if (period_y > 0) vy = wrap_f(vy, (float)period_y);
        // M * v: M = [[0,1,1],[1,0,1],[1,1,0]] → (vy+vz, vx+vz, vx+vy)
        float wx = vy + v.z;
        float wy = vx + v.z;
        float wz = vx + vy;
        int ix = (int)std::floor(wx + 0.5f);
        int iy = (int)std::floor(wy + 0.5f);
        int iz = (int)std::floor(wz + 0.5f);
        return pcg3d((uint32_t)ix, (uint32_t)iy, (uint32_t)iz);
    };

    uvec3 h0 = wrap_and_hash(v0);
    uvec3 h1 = wrap_and_hash(v1);
    uvec3 h2 = wrap_and_hash(v2);
    uvec3 h3 = wrap_and_hash(v3);

    // Generate gradient direction from hash
    float hf0 = (float)(h0.x & 0xFFFFu) * (1.0f / 65536.0f);
    float hf1 = (float)(h1.x & 0xFFFFu) * (1.0f / 65536.0f);
    float hf2 = (float)(h2.x & 0xFFFFu) * (1.0f / 65536.0f);
    float hf3 = (float)(h3.x & 0xFFFFu) * (1.0f / 65536.0f);

    float psi0 = hf0 * TS_HASH_TO_ANGLE + alpha;
    float psi1 = hf1 * TS_HASH_TO_ANGLE + alpha;
    float psi2 = hf2 * TS_HASH_TO_ANGLE + alpha;
    float psi3 = hf3 * TS_HASH_TO_ANGLE + alpha;

    // 2D gradients (xy only)
    Vec3 g0 = {std::cos(psi0), std::sin(psi0), 0.0f};
    Vec3 g1 = {std::cos(psi1), std::sin(psi1), 0.0f};
    Vec3 g2 = {std::cos(psi2), std::sin(psi2), 0.0f};
    Vec3 g3 = {std::cos(psi3), std::sin(psi3), 0.0f};

    auto dot3 = [](Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; };

    // Radial decay kernel (3D: 0.5 support radius)
    float d0 = dot3(x0, x0), d1 = dot3(x1, x1), d2 = dot3(x2, x2), d3 = dot3(x3_, x3_);
    float w[4] = {
        std::max(0.5f - d0, 0.0f),
        std::max(0.5f - d1, 0.0f),
        std::max(0.5f - d2, 0.0f),
        std::max(0.5f - d3, 0.0f)
    };
    float w2[4], w3[4];
    for (int i = 0; i < 4; i++) { w2[i] = w[i]*w[i]; w3[i] = w2[i]*w[i]; }

    Vec3 xs[4] = {x0, x1, x2, x3_};
    Vec3 gs[4] = {g0, g1, g2, g3};
    float gdotx[4];
    for (int i = 0; i < 4; i++) gdotx[i] = dot3(gs[i], xs[i]);

    float n = 0.0f;
    for (int i = 0; i < 4; i++) n += w3[i] * gdotx[i];

    // Gradient
    float dn0x = w3[0]*g0.x + (-6.0f*w2[0]*gdotx[0])*x0.x;
    float dn0y = w3[0]*g0.y + (-6.0f*w2[0]*gdotx[0])*x0.y;
    float dn1x = w3[1]*g1.x + (-6.0f*w2[1]*gdotx[1])*x1.x;
    float dn1y = w3[1]*g1.y + (-6.0f*w2[1]*gdotx[1])*x1.y;
    float dn2x = w3[2]*g2.x + (-6.0f*w2[2]*gdotx[2])*x2.x;
    float dn2y = w3[2]*g2.y + (-6.0f*w2[2]*gdotx[2])*x2.y;
    float dn3x = w3[3]*g3.x + (-6.0f*w2[3]*gdotx[3])*x3_.x;
    float dn3y = w3[3]*g3.y + (-6.0f*w2[3]*gdotx[3])*x3_.y;

    SimplexResult r;
    r.value = TS_SIMPLEX_NORM_3D_CPU * n;
    r.gx = TS_SIMPLEX_NORM_3D_CPU * (dn0x + dn1x + dn2x + dn3x);
    r.gy = TS_SIMPLEX_NORM_3D_CPU * (dn0y + dn1y + dn2y + dn3y);
    return r;
}

static float eval_simplex_3d(float uv_x, float uv_y, int period, float alpha) {
    int iper = std::max(period, 1);
    float p_x = uv_x * (float)iper;
    float p_y = uv_y * (float)iper;
    return ts_simplex_tile_3d_cpu(p_x, p_y, iper, iper, alpha).value;
}

// ============================================================================
// 3D SIMPLEX GRID TESTS: Verify tiling at ALL integer periods
// ============================================================================

TEST(Simplex3D, HorizontalEdgeMatch_AllPeriods) {
    printf("\n=== 3D SIMPLEX HORIZONTAL EDGE — ALL integer periods ===\n");
    printf("%-8s | %-12s | %-12s | %-10s\n", "Period", "MaxDiff_R", "MaxDiff_G", "TILES?");
    printf("---------|-------------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;

    for (int period = 2; period <= 10; period++) {
        float max_diff_r = 0.0f, max_diff_g = 0.0f;
        for (int y = 0; y < RES; y++) {
            float uv_y = (float)y / (float)(RES - 1);
            float vl_r = eval_simplex_3d(0.0f, uv_y, period, ALPHA);
            float vr_r = eval_simplex_3d(1.0f, uv_y, period, ALPHA);
            float vl_g = eval_simplex_3d(0.0f, uv_y, period, ALPHA + 1.0f);
            float vr_g = eval_simplex_3d(1.0f, uv_y, period, ALPHA + 1.0f);
            max_diff_r = std::max(max_diff_r, std::abs(vl_r - vr_r));
            max_diff_g = std::max(max_diff_g, std::abs(vl_g - vr_g));
        }
        bool tiles = (max_diff_r < 0.001f && max_diff_g < 0.001f);
        printf("%-8d | %-12.8f | %-12.8f | %s\n",
               period, max_diff_r, max_diff_g, tiles ? "YES" : "NO ***");
        EXPECT_TRUE(tiles) << "3D simplex HORIZONTAL tiling FAILED at period " << period;
    }
}

TEST(Simplex3D, VerticalEdgeMatch_AllPeriods) {
    printf("\n=== 3D SIMPLEX VERTICAL EDGE — ALL integer periods ===\n");
    printf("%-8s | %-12s | %-12s | %-10s\n", "Period", "MaxDiff_R", "MaxDiff_G", "TILES?");
    printf("---------|-------------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;

    for (int period = 2; period <= 10; period++) {
        float max_diff_r = 0.0f, max_diff_g = 0.0f;
        for (int x = 0; x < RES; x++) {
            float uv_x = (float)x / (float)(RES - 1);
            float vt_r = eval_simplex_3d(uv_x, 0.0f, period, ALPHA);
            float vb_r = eval_simplex_3d(uv_x, 1.0f, period, ALPHA);
            float vt_g = eval_simplex_3d(uv_x, 0.0f, period, ALPHA + 1.0f);
            float vb_g = eval_simplex_3d(uv_x, 1.0f, period, ALPHA + 1.0f);
            max_diff_r = std::max(max_diff_r, std::abs(vt_r - vb_r));
            max_diff_g = std::max(max_diff_g, std::abs(vt_g - vb_g));
        }
        bool tiles = (max_diff_r < 0.001f && max_diff_g < 0.001f);
        printf("%-8d | %-12.8f | %-12.8f | %s\n",
               period, max_diff_r, max_diff_g, tiles ? "YES" : "NO ***");
        EXPECT_TRUE(tiles) << "3D simplex VERTICAL tiling FAILED at period " << period;
    }
}

TEST(Simplex3D, BothEdges_AllPeriods) {
    printf("\n=== 3D SIMPLEX BOTH EDGES (X+Y) — ALL integer periods ===\n");
    printf("%-8s | %-12s | %-10s\n", "Period", "MaxDiff", "TILES?");
    printf("---------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;

    for (int period = 2; period <= 10; period++) {
        float max_diff = 0.0f;
        for (int y = 0; y < RES; y++) {
            float uv_y = (float)y / (float)(RES - 1);
            float vl = eval_simplex_3d(0.0f, uv_y, period, ALPHA);
            float vr = eval_simplex_3d(1.0f, uv_y, period, ALPHA);
            max_diff = std::max(max_diff, std::abs(vl - vr));
        }
        for (int x = 0; x < RES; x++) {
            float uv_x = (float)x / (float)(RES - 1);
            float vt = eval_simplex_3d(uv_x, 0.0f, period, ALPHA);
            float vb = eval_simplex_3d(uv_x, 1.0f, period, ALPHA);
            max_diff = std::max(max_diff, std::abs(vt - vb));
        }
        bool tiles = (max_diff < 0.001f);
        printf("%-8d | %-12.8f | %s\n", period, max_diff, tiles ? "YES" : "NO ***");
        EXPECT_TRUE(tiles) << "3D simplex BOTH-EDGE tiling FAILED at period " << period;
    }
}

// ============================================================================
// 3D SIMPLEX: Gradient continuity across the seam boundary
// Checks that the slope from pixel(res-2)->(res-1) matches (res-1)->(0+1)
// This is what causes faint displacement lines even when values match.
// ============================================================================
TEST(Simplex3D, GradientContinuity_SeamBoundary) {
    printf("\n=== 3D SIMPLEX GRADIENT CONTINUITY ACROSS SEAM ===\n");
    printf("Checks that the slope is uniform across pixel(res-2)->(res-1)->(0)->(1)\n");
    printf("%-8s | %-12s | %-12s | %-10s\n", "Period", "MaxGradJump", "Threshold", "SMOOTH?");
    printf("---------|-------------|-------------|----------\n");

    const int RES = 256;
    const float ALPHA = 0.0f;

    for (int period = 2; period <= 10; period++) {
        float max_grad_jump = 0.0f;

        // Check horizontal seam: for each row, compare gradients left vs right of seam
        for (int y = 0; y < RES; y++) {
            float uv_y = (float)y / (float)(RES - 1);
            // Three consecutive values across the seam:
            // pixel(res-2) -> pixel(res-1) -> pixel(0) of next tile (= pixel(0))
            float v_prev = eval_simplex_3d((float)(RES - 2) / (float)(RES - 1), uv_y, period, ALPHA);
            float v_last = eval_simplex_3d(1.0f, uv_y, period, ALPHA);
            float v_first = eval_simplex_3d(0.0f, uv_y, period, ALPHA);
            float v_next = eval_simplex_3d(1.0f / (float)(RES - 1), uv_y, period, ALPHA);

            // Gradient before seam: (v_last - v_prev) * (RES-1)
            float grad_before = (v_last - v_prev) * (float)(RES - 1);
            // Gradient after seam: (v_next - v_first) * (RES-1)  (same pixel spacing)
            float grad_after = (v_next - v_first) * (float)(RES - 1);

            float jump = std::abs(grad_after - grad_before);
            max_grad_jump = std::max(max_grad_jump, jump);
        }

        // Same for vertical seam
        for (int x = 0; x < RES; x++) {
            float uv_x = (float)x / (float)(RES - 1);
            float v_prev = eval_simplex_3d(uv_x, (float)(RES - 2) / (float)(RES - 1), period, ALPHA);
            float v_last = eval_simplex_3d(uv_x, 1.0f, period, ALPHA);
            float v_first = eval_simplex_3d(uv_x, 0.0f, period, ALPHA);
            float v_next = eval_simplex_3d(uv_x, 1.0f / (float)(RES - 1), period, ALPHA);

            float grad_before = (v_last - v_prev) * (float)(RES - 1);
            float grad_after = (v_next - v_first) * (float)(RES - 1);

            float jump = std::abs(grad_after - grad_before);
            max_grad_jump = std::max(max_grad_jump, jump);
        }

        bool smooth = (max_grad_jump < 0.5f);
        printf("%-8d | %-12.6f | %-12.6f | %s\n",
               period, max_grad_jump, 0.5f, smooth ? "YES" : "NO ***");
    }
}

// ============================================================================
// 3D SIMPLEX: Print actual pixel values at the seam boundary
// Shows the raw values so the user can see the discontinuity.
// ============================================================================
TEST(Simplex3D, PrintSeamValues) {
    printf("\n=== 3D SIMPLEX RAW VALUES AT SEAM BOUNDARY ===\n");
    printf("Shows pixel(res-2), pixel(res-1), pixel(0) for a few rows\n");

    const int RES = 256;
    const float ALPHA = 0.0f;
    const int PERIOD = 3;

    printf("\n--- Period %d, HORIZONTAL seam (left edge) ---\n", PERIOD);
    printf("%-6s | %-14s | %-14s | %-14s | %-14s | %-12s\n",
           "y", "px[res-2]", "px[res-1]", "px[0]", "px[1]", "grad_jump");
    printf("-------|--------------|--------------|--------------|--------------|------------\n");
    for (int y = 0; y < 16; y++) {
        float uv_y = (float)y / (float)(RES - 1);
        float v_prev = eval_simplex_3d((float)(RES - 2) / (float)(RES - 1), uv_y, PERIOD, ALPHA);
        float v_last = eval_simplex_3d(1.0f, uv_y, PERIOD, ALPHA);
        float v_first = eval_simplex_3d(0.0f, uv_y, PERIOD, ALPHA);
        float v_next = eval_simplex_3d(1.0f / (float)(RES - 1), uv_y, PERIOD, ALPHA);

        float grad_before = (v_last - v_prev) * (float)(RES - 1);
        float grad_after = (v_next - v_first) * (float)(RES - 1);
        float jump = grad_after - grad_before;

        printf("%-6d | %+14.10f | %+14.10f | %+14.10f | %+14.10f | %+.8f\n",
               y, v_prev, v_last, v_first, v_next, jump);
    }

    printf("\n--- Period %d, VERTICAL seam (top edge) ---\n", PERIOD);
    printf("%-6s | %-14s | %-14s | %-14s | %-14s | %-12s\n",
           "x", "px[res-2]", "px[res-1]", "px[0]", "px[1]", "grad_jump");
    printf("-------|--------------|--------------|--------------|--------------|------------\n");
    for (int x = 0; x < 16; x++) {
        float uv_x = (float)x / (float)(RES - 1);
        float v_prev = eval_simplex_3d(uv_x, (float)(RES - 2) / (float)(RES - 1), PERIOD, ALPHA);
        float v_last = eval_simplex_3d(uv_x, 1.0f, PERIOD, ALPHA);
        float v_first = eval_simplex_3d(uv_x, 0.0f, PERIOD, ALPHA);
        float v_next = eval_simplex_3d(uv_x, 1.0f / (float)(RES - 1), PERIOD, ALPHA);

        float grad_before = (v_last - v_prev) * (float)(RES - 1);
        float grad_after = (v_next - v_first) * (float)(RES - 1);
        float jump = grad_after - grad_before;

        printf("%-6d | %+14.10f | %+14.10f | %+14.10f | %+14.10f | %+.8f\n",
               x, v_prev, v_last, v_first, v_next, jump);
    }
}

// ============================================================================
// 3D SIMPLEX: Is the gradient jump at the seam WORSE than at random interior
// cell boundaries? If not, it's just normal simplex cell structure, not a bug.
// ============================================================================
TEST(Simplex3D, GradientJump_SeamVsInterior) {
    printf("\n=== 3D SIMPLEX: Gradient jump at SEAM vs INTERIOR cell boundary ===\n");
    printf("If seam jump ≈ interior jump, it's inherent simplex behavior, not a bug\n\n");
    printf("%-8s | %-12s | %-12s | %-10s\n", "Period", "SeamJump", "InteriorJump", "VERDICT");
    printf("---------|-------------|-------------|----------\n");

    const float ALPHA = 0.0f;

    for (int period = 2; period <= 10; period++) {
        float max_seam_jump = 0.0f;
        float max_interior_jump = 0.0f;

        // Measure gradient jump at the seam (uv=0 vs uv=1 boundary)
        for (int i = 0; i < 256; i++) {
            float dp = (float)period / 255.0f; // pixel step in p-space

            // Use the actual seam: last pixel = p=period, first pixel = p=0
            float seam_prev = ts_simplex_tile_3d_cpu((float)period - 2.0f*dp, 0.0f, period, period, ALPHA).value;
            float seam_last = ts_simplex_tile_3d_cpu((float)period - dp, 0.0f, period, period, ALPHA).value;
            float seam_first = ts_simplex_tile_3d_cpu(0.0f, 0.0f, period, period, ALPHA).value;
            float seam_next  = ts_simplex_tile_3d_cpu(dp, 0.0f, period, period, ALPHA).value;

            float grad_before_seam = (seam_last - seam_prev) / dp;
            float grad_after_seam  = (seam_next - seam_first) / dp;
            max_seam_jump = std::max(max_seam_jump, std::abs(grad_after_seam - grad_before_seam));

            // Interior: pick a point in the middle of the texture, step through p-space
            float mid_p = (float)period * 0.5f;
            float int_prev = ts_simplex_tile_3d_cpu(mid_p - 2.0f*dp, 0.0f, period, period, ALPHA).value;
            float int_last = ts_simplex_tile_3d_cpu(mid_p - dp, 0.0f, period, period, ALPHA).value;
            float int_first = ts_simplex_tile_3d_cpu(mid_p, 0.0f, period, period, ALPHA).value;
            float int_next  = ts_simplex_tile_3d_cpu(mid_p + dp, 0.0f, period, period, ALPHA).value;

            float grad_before_int = (int_last - int_prev) / dp;
            float grad_after_int  = (int_next - int_first) / dp;
            max_interior_jump = std::max(max_interior_jump, std::abs(grad_after_int - grad_before_int));
        }

        const char* verdict = (max_seam_jump < max_interior_jump * 2.0f)
            ? "NORMAL (same as interior)"
            : "WORSE (wrapping artifact!)";
        printf("%-8d | %-12.6f | %-12.6f | %s\n",
               period, max_seam_jump, max_interior_jump, verdict);
    }
}
TEST(Simplex3D, WrapConsistency) {
    printf("\n=== 3D SIMPLEX: Does wrapping produce identical results? ===\n");
    printf("Compares noise(p=0) vs noise(p=period) directly\n\n");

    const float ALPHA = 0.0f;
    const int RES = 256;

    for (int period = 2; period <= 10; period++) {
        float max_diff_val = 0.0f;
        float max_diff_gx = 0.0f;
        float max_diff_gy = 0.0f;

        for (int y = 0; y < RES; y++) {
            for (int x = 0; x < RES; x++) {
                float uv_x = (float)x / (float)(RES - 1);
                float uv_y = (float)y / (float)(RES - 1);
                float p_x = uv_x * (float)period;
                float p_y = uv_y * (float)period;

                auto r0 = ts_simplex_tile_3d_cpu(p_x, p_y, period, period, ALPHA);
                // Shift by exactly one period — should wrap to same value
                auto r1 = ts_simplex_tile_3d_cpu(p_x + (float)period, p_y, period, period, ALPHA);
                auto r2 = ts_simplex_tile_3d_cpu(p_x, p_y + (float)period, period, period, ALPHA);

                max_diff_val = std::max(max_diff_val, std::abs(r0.value - r1.value));
                max_diff_val = std::max(max_diff_val, std::abs(r0.value - r2.value));
                max_diff_gx = std::max(max_diff_gx, std::abs(r0.gx - r1.gx));
                max_diff_gx = std::max(max_diff_gx, std::abs(r0.gx - r2.gx));
                max_diff_gy = std::max(max_diff_gy, std::abs(r0.gy - r1.gy));
                max_diff_gy = std::max(max_diff_gy, std::abs(r0.gy - r2.gy));
            }
        }
        printf("Period %2d: val_diff=%.10f  gx_diff=%.10f  gy_diff=%.10f %s\n",
               period, max_diff_val, max_diff_gx, max_diff_gy,
               (max_diff_val < 1e-6f && max_diff_gx < 1e-6f && max_diff_gy < 1e-6f)
               ? "PERFECT" : "MISMATCH ***");
    }
}

TEST(Simplex3D, Period3DiffersFromPeriod4) {
    printf("\n=== 3D SIMPLEX: period 3 vs 4 (must be DIFFERENT) ===\n");
    const int RES = 256;
    const float ALPHA = 0.0f;
    for (auto& [odd, even] : std::vector<std::pair<int,int>>{{3,4},{5,6},{7,8}}) {
        float max_diff = 0.0f;
        for (int y = 0; y < RES; y++) {
            for (int x = 0; x < RES; x++) {
                float uv_x = (float)x / (float)(RES - 1);
                float uv_y = (float)y / (float)(RES - 1);
                float vo = eval_simplex_3d(uv_x, uv_y, odd, ALPHA);
                float ve = eval_simplex_3d(uv_x, uv_y, even, ALPHA);
                max_diff = std::max(max_diff, std::abs(vo - ve));
            }
        }
        bool different = (max_diff > 0.001f);
        printf("Period %d vs %d: max_diff=%.8f %s\n",
               odd, even, max_diff, different ? "DIFFERENT (good!)" : "IDENTICAL (bad!)");
        EXPECT_TRUE(different) << "3D simplex period " << odd << " vs " << even << " are IDENTICAL";
    }
}
