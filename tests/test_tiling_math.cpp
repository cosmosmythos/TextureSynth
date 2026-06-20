#include <cmath>
#include <cstdio>
#include <cstdint>
#include <algorithm>

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
            float d2 = dx * dx + dy * dy;
            if (d2 < f1) f1 = d2;
        }
    }
    return std::clamp(std::sqrt(f1) * 0.70710678f, 0.0f, 1.0f);
}

static float worley_cpu_fract(float ux, float uy, int per, float jitter, uint32_t seed) {
    float fx = ux - std::floor(ux);
    float fy = uy - std::floor(uy);
    float px = fx * (float)per;
    float py = fy * (float)per;
    return worley_cpu(px, py, per, jitter, seed);
}

int main() {
    const int res = 256;
    const float jitter = 1.0f;
    const uint32_t seed = 0;

    printf("=== TEST 1: Periodicity proof  f(p) == f(p+per) ===\n");
    for (int per = 2; per <= 32; ++per) {
        int mismatches = 0;
        for (int trial = 0; trial < 200; ++trial) {
            float px = (float)trial * 0.137f + 0.5f;
            float py = (float)(trial * 7) * 0.091f + 0.3f;
            float v1 = worley_cpu(px, py, per, jitter, seed);
            float v2 = worley_cpu(px + (float)per, py, per, jitter, seed);
            if (v1 != v2) mismatches++;
        }
        printf("  per=%2d  mismatches=%d/200  %s\n", per, mismatches,
               mismatches == 0 ? "OK" : "BROKEN");
    }

    printf("\n=== TEST 2: CPU seam (no fract)  LR=|f(0,y)-f(255*per/256,y)| ===\n");
    printf("%-6s %-6s %-12s %-12s\n", "per", "pow2?", "LR_max", "TB_max");
    printf("------ ------ ------------ ------------\n");
    for (int per = 2; per <= 32; ++per) {
        bool is_pow2 = (per & (per - 1)) == 0;
        float lr_max = 0, tb_max = 0;
        for (int y = 0; y < res; ++y) {
            float pL = worley_cpu(0.0f, (float)y * per / (float)res, per, jitter, seed);
            float pR = worley_cpu((float)(res - 1) * per / (float)res, (float)y * per / (float)res, per, jitter, seed);
            lr_max = std::max(lr_max, std::abs(pL - pR));
        }
        for (int x = 0; x < res; ++x) {
            float pT = worley_cpu((float)x * per / (float)res, 0.0f, per, jitter, seed);
            float pB = worley_cpu((float)x * per / (float)res, (float)(res - 1) * per / (float)res, per, jitter, seed);
            tb_max = std::max(tb_max, std::abs(pT - pB));
        }
        printf("%-6d %-6s %-12.3f %-12.3f\n", per, is_pow2 ? "YES" : "no", lr_max, tb_max);
    }

    printf("\n=== TEST 3: CPU seam (WITH fract)  uv=coord/res, p=fract(uv)*per ===\n");
    printf("%-6s %-6s %-12s %-12s\n", "per", "pow2?", "LR_max", "TB_max");
    printf("------ ------ ------------ ------------\n");
    for (int per = 2; per <= 32; ++per) {
        bool is_pow2 = (per & (per - 1)) == 0;
        float lr_max = 0, tb_max = 0;
        for (int y = 0; y < res; ++y) {
            float uvLy = (float)y / (float)res;
            float uvRy = (float)y / (float)res;
            float pL = worley_cpu_fract(0.0f / (float)res, uvLy, per, jitter, seed);
            float pR = worley_cpu_fract((float)(res - 1) / (float)res, uvRy, per, jitter, seed);
            lr_max = std::max(lr_max, std::abs(pL - pR));
        }
        for (int x = 0; x < res; ++x) {
            float uvTx = (float)x / (float)res;
            float pT = worley_cpu_fract(uvTx, 0.0f / (float)res, per, jitter, seed);
            float pB = worley_cpu_fract(uvTx, (float)(res - 1) / (float)res, per, jitter, seed);
            tb_max = std::max(tb_max, std::abs(pT - pB));
        }
        printf("%-6d %-6s %-12.3f %-12.3f\n", per, is_pow2 ? "YES" : "no", lr_max, tb_max);
    }

    printf("\n=== TEST 4: Exact periodicity at integer lattice  f(0)==f(per) ===\n");
    for (int per = 2; per <= 32; ++per) {
        float v0 = worley_cpu(0.0f, 0.0f, per, jitter, seed);
        float vp = worley_cpu((float)per, 0.0f, per, jitter, seed);
        printf("  per=%2d  f(0)=%.10f  f(%d)=%.10f  %s\n",
               per, v0, per, vp, v0 == vp ? "MATCH" : "MISMATCH");
    }

    return 0;
}
