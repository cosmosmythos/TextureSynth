#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform texture2D u_sampled[];
layout(set = 0, binding = 1, rgba32f) writeonly uniform image2D u_storage[];

layout(set = 0, binding = 2) uniform sampler samp_repeat;
layout(set = 0, binding = 3) uniform sampler samp_clamp;
layout(set = 0, binding = 4) uniform sampler samp_mirror;
layout(set = 0, binding = 5, std430) readonly buffer NodeParams { float v[]; } node_params[];
layout(push_constant, std430) uniform PC {
    uint resolution_x, resolution_y, seed;
    float time;
    uint out_storage_slots[4];
    uint param_base_slot;
    uint input_count;
    uint param_ring_idx;
    uint in_sampled_slots[8];
} pc;

struct TSTexture { int slot; vec2 inv_size; };

vec4 ts_sample(int local, vec2 uv) {
    return texture(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[local])], samp_repeat), uv);
}
vec4 ts_sample_lod(int local, vec2 uv, float lod) {
    return textureLod(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[local])], samp_repeat), uv, lod);
}
ivec2 ts_size(int local) {
    return textureSize(u_sampled[nonuniformEXT(pc.in_sampled_slots[local])], 0);
}
vec4  Sample      (TSTexture t, vec2 uv)            { return ts_sample(t.slot, uv); }
vec4  SampleLevel (TSTexture t, vec2 uv, float lod) { return ts_sample_lod(t.slot, uv, lod); }
ivec2 GetSize     (TSTexture t)                     { return ts_size(t.slot); }
vec2  GetTexelSize(TSTexture t)                     { return t.inv_size; }




vec4 _fmt_mono(vec4 v) { return vec4(v.x, 0.0, 0.0, 1.0); }
vec4 _fmt_uv(vec4 v) { return vec4(v.xy, 0.0, 1.0); }
vec4 _fmt_rgb(vec4 v) { return vec4(v.rgb, 1.0); }
vec4 _fmt_id(vec4 v) { return v; }
vec4 _fmt_metadata(vec4 v) { return v; }


#ifndef TS_REMAP
#define TS_REMAP

float ts_remap(float v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return out_min;
    return out_min + (out_max - out_min) * (v - in_min) / denom;
}

vec2 ts_remap(vec2 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec2(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec3 ts_remap(vec3 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec3(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec4 ts_remap(vec4 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec4(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

#endif

#ifndef TS_NOISE_COMMON
#define TS_NOISE_COMMON

//PCG INTEGER HASH
uvec2 ts_pcg2d(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    return v;
}

uvec3 ts_pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

uvec4 ts_pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    uint x = v.x + v.y * v.w;
    uint y = v.y + v.z * x;
    uint z = v.z + v.x * y;
    uint w = v.w + y * z;
    v = uvec4(x, y, z, w) ^ (uvec4(x, y, z, w) >> 16u);
    x = v.x + v.y * v.w;
    y = v.y + v.z * x;
    z = v.z + v.x * y;
    w = v.w + y * z;
    return uvec4(x, y, z, w);
}



// LATTICE HASHES  (pcg3d-based — seed rides natively in .z)
const float TS_HASH_TO_ANGLE = 21.548;

float ts_hash2(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return float(h.x & 0xFFFFu) * (1.0 / 65536.0);
}

vec4 ts_hash2_quad(ivec2 c00, uint seed) {
    uvec3 h00 = ts_pcg3d(uvec3(uvec2(c00),                seed));
    uvec3 h10 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(1u, 0u), seed));
    uvec3 h01 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(0u, 1u), seed));
    uvec3 h11 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(1u, 1u), seed));
    return vec4(h00.x & 0xFFFFu, h10.x & 0xFFFFu, h01.x & 0xFFFFu, h11.x & 0xFFFFu)
         * (1.0 / 65536.0);
}

vec2 ts_hash2_vec2(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return vec2(h.x & 0xFFFFu, h.y & 0xFFFFu) * (1.0 / 65536.0);
}

vec3 ts_hash2_vec3(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return vec3(h.x & 0xFFFFu, h.y & 0xFFFFu, h.z & 0xFFFFu) * (1.0 / 65536.0);
}

vec2 ts_grad8(float hash01) {
    float a = floor(hash01 * 8.0) * 0.78539816339; // π/4
    return vec2(cos(a), sin(a));
}

vec2 ts_grad16(float hash01) {
    float a = floor(hash01 * 16.0) * 0.39269908170; // π/8
    return vec2(cos(a), sin(a));
}

vec2 ts_grad_continuous(float hash01) {
    float a = hash01 * TS_HASH_TO_ANGLE;
    return vec2(cos(a), sin(a));
}


// SMOOTHSTEP
float ts_quintic(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

vec2 ts_quintic(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float ts_quintic_d(float t) {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

vec2 ts_quintic_d(vec2 t) {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}


// PERIOD
int ts_period_int(float p) {
    return max(int(round(p)), 1);
}

int ts_period_pow2(float p) {
    int v = max(int(round(p)), 1);
    float l = floor(log2(float(v)));
    int lo = int(l);
    int pow2_lo = 1 << lo;
    int pow2_hi = pow2_lo * 2;
    return (v - pow2_lo <= pow2_hi - v) ? pow2_lo : pow2_hi;
}

int ts_period_even(float p) {
    return max(int(round(p * 0.5)) * 2, 2);
}


int ts_wrap(int v, int per) {
    return int(mod(float(v), float(per)));
}

ivec2 ts_wrap2(ivec2 v, int per) {
    return ivec2(ts_wrap(v.x, per), ts_wrap(v.y, per));
}


float ts_value_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,               per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0), per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1), per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1), per);

    // Corner values: unit hash → [-1, 1]
    float v00 = ts_hash2(c00, seed) * 2.0 - 1.0;
    float v10 = ts_hash2(c10, seed) * 2.0 - 1.0;
    float v01 = ts_hash2(c01, seed) * 2.0 - 1.0;
    float v11 = ts_hash2(c11, seed) * 2.0 - 1.0;

    vec2 u  = ts_quintic(pf);
    vec2 du = ts_quintic_d(pf);

    float a = mix(v00, v10, u.x);
    float b = mix(v01, v11, u.x);
    float n = mix(a, b, u.y);

    // ∂n/∂x = du.x × [(v10-v00)(1-u.y) + (v11-v01)u.y]
    // ∂n/∂y = du.y × [(v01-v00)(1-u.x) + (v11-v10)u.x]
    grad.x = du.x * mix(v10 - v00, v11 - v01, u.y);
    grad.y = du.y * mix(v01 - v00, v11 - v10, u.x);

    return n;
}

float ts_value_tile(vec2 p, int per, uint seed) {
    vec2 g;
    return ts_value_tile(p, per, seed, g);
}

const float TS_PERLIN_NORM = 1.3393713;   // = 1.0 / 0.74682413

float ts_perlin_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,               per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0), per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1), per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1), per);

    vec2 g00 = ts_grad8(ts_hash2(c00, seed));
    vec2 g10 = ts_grad8(ts_hash2(c10, seed));
    vec2 g01 = ts_grad8(ts_hash2(c01, seed));
    vec2 g11 = ts_grad8(ts_hash2(c11, seed));

    vec2 d00 = pf;
    vec2 d10 = pf - vec2(1.0, 0.0);
    vec2 d01 = pf - vec2(0.0, 1.0);
    vec2 d11 = pf - vec2(1.0, 1.0);

    float n00 = dot(g00, d00);
    float n10 = dot(g10, d10);
    float n01 = dot(g01, d01);
    float n11 = dot(g11, d11);

    vec2 u  = ts_quintic(pf);
    vec2 du = ts_quintic_d(pf);

    float a = mix(n00, n10, u.x);
    float b = mix(n01, n11, u.x);
    float n = mix(a, b, u.y);

    // Analytical gradient (chain rule through bilinear blend + dot products)
    vec2 ga = mix(g00, g10, u.x) + du.x * (n10 - n00) * vec2(1.0, 0.0);
    vec2 gb = mix(g01, g11, u.x) + du.x * (n11 - n01) * vec2(1.0, 0.0);
    grad   = mix(ga, gb, u.y) + du.y * (b - a) * vec2(0.0, 1.0);
    grad  *= TS_PERLIN_NORM;

    return n * TS_PERLIN_NORM;
}

float ts_perlin_tile(vec2 p, int per, uint seed) {
    vec2 g;
    return ts_perlin_tile(p, per, seed, g);
}

const float TS_SIMPLEX_NORM_3D = 39.5;

// 3D axis-aligned simplex grid (tetrahedral lattice) evaluated on a 2D slice
// (z=0). Tiles seamlessly at ANY integer period in x and y.
// Based on Gustavson & McEwan, JCGT 11(1), 2022 — psrdnoise3.
float ts_simplex_tile(vec2 x, ivec2 period, float alpha, out vec2 gradient) {
    // Promote to 3D: z=0 slice, no Z wrapping
    vec3 x3 = vec3(x.x, x.y, 0.0);

    // Axis-aligned simplex grid transform: uvw = M * x
    // M = [[0,1,1],[1,0,1],[1,1,0]]
    vec3 uvw = vec3(x3.y + x3.z, x3.x + x3.z, x3.x + x3.y);

    vec3 i0 = floor(uvw);
    vec3 f0 = fract(uvw);

    // Rank-order fractional parts to determine tetrahedron corners
    vec3 g_ = step(f0.xyx, f0.yzz);
    vec3 l_ = 1.0 - g_;
    vec3 g  = vec3(l_.z, g_.xy);
    vec3 l  = vec3(l_.xy, g_.z);
    vec3 o1 = min(g, l);
    vec3 o2 = max(g, l);

    vec3 i1 = i0 + o1;
    vec3 i2 = i0 + o2;
    vec3 i3 = i0 + vec3(1.0);

    // Inverse transform Mi = [[-0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5]]
    vec3 v0 = vec3(-0.5*i0.x + 0.5*i0.y + 0.5*i0.z,
                     0.5*i0.x - 0.5*i0.y + 0.5*i0.z,
                     0.5*i0.x + 0.5*i0.y - 0.5*i0.z);
    vec3 v1 = vec3(-0.5*i1.x + 0.5*i1.y + 0.5*i1.z,
                     0.5*i1.x - 0.5*i1.y + 0.5*i1.z,
                     0.5*i1.x + 0.5*i1.y - 0.5*i1.z);
    vec3 v2 = vec3(-0.5*i2.x + 0.5*i2.y + 0.5*i2.z,
                     0.5*i2.x - 0.5*i2.y + 0.5*i2.z,
                     0.5*i2.x + 0.5*i2.y - 0.5*i2.z);
    vec3 v3 = vec3(-0.5*i3.x + 0.5*i3.y + 0.5*i3.z,
                     0.5*i3.x - 0.5*i3.y + 0.5*i3.z,
                     0.5*i3.x + 0.5*i3.y - 0.5*i3.z);

    // Displacement vectors (in texture space)
    vec3 x0 = x3 - v0;
    vec3 x1 = x3 - v1;
    vec3 x2 = x3 - v2;
    vec3 x3_ = x3 - v3;

    // Wrap corner positions and transform back to simplex space
    if (period.x > 0 || period.y > 0) {
        vec4 vx = vec4(v0.x, v1.x, v2.x, v3.x);
        vec4 vy = vec4(v0.y, v1.y, v2.y, v3.y);
        vec4 vz = vec4(v0.z, v1.z, v2.z, v3.z);

        if (period.x > 0) vx = mod(vx, float(period.x));
        if (period.y > 0) vy = mod(vy, float(period.y));

        // Transform wrapped texture-space positions back to simplex space
        // M = [[0,1,1],[1,0,1],[1,1,0]]: M*v = (v.y+v.z, v.x+v.z, v.x+v.y)
        vec3 w0 = vec3(vy.x + vz.x, vx.x + vz.x, vx.x + vy.x);
        vec3 w1 = vec3(vy.y + vz.y, vx.y + vz.y, vx.y + vy.y);
        vec3 w2 = vec3(vy.z + vz.z, vx.z + vz.z, vx.z + vy.z);
        vec3 w3 = vec3(vy.w + vz.w, vx.w + vz.w, vx.w + vy.w);

        // Fix rounding errors
        i0 = floor(w0 + 0.5);
        i1 = floor(w1 + 0.5);
        i2 = floor(w2 + 0.5);
        i3 = floor(w3 + 0.5);
    }

    // Hash each corner via pcg3d
    uvec3 h0 = ts_pcg3d(uvec3(uint(i0.x), uint(i0.y), uint(i0.z)));
    uvec3 h1 = ts_pcg3d(uvec3(uint(i1.x), uint(i1.y), uint(i1.z)));
    uvec3 h2 = ts_pcg3d(uvec3(uint(i2.x), uint(i2.y), uint(i2.z)));
    uvec3 h3 = ts_pcg3d(uvec3(uint(i3.x), uint(i3.y), uint(i3.z)));

    // Generate gradient direction from hash
    float hf0 = float(h0.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf1 = float(h1.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf2 = float(h2.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf3 = float(h3.x & 0xFFFFu) * (1.0 / 65536.0);

    vec4 psi = vec4(hf0, hf1, hf2, hf3) * TS_HASH_TO_ANGLE + alpha;
    // 2D gradients (xy only — z=0 slice)
    vec4 gx = cos(psi);
    vec4 gy = sin(psi);

    vec3 g0 = vec3(gx.x, gy.x, 0.0);
    vec3 g1 = vec3(gx.y, gy.y, 0.0);
    vec3 g2 = vec3(gx.z, gy.z, 0.0);
    vec3 g3 = vec3(gx.w, gy.w, 0.0);

    // Radial decay kernel (3D: 0.5 support radius)
    vec4 w  = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3_,x3_)), 0.0);
    vec4 w2 = w * w;
    vec4 w3 = w2 * w;

    vec4 gdotx = vec4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3_));
    float n = dot(w3, gdotx);

    // Analytical gradient
    vec4 dw  = -6.0 * w2 * gdotx;
    vec3 dn0 = w3.x * g0 + dw.x * x0;
    vec3 dn1 = w3.y * g1 + dw.y * x1;
    vec3 dn2 = w3.z * g2 + dw.z * x2;
    vec3 dn3 = w3.w * g3 + dw.w * x3_;

    vec3 grad3 = TS_SIMPLEX_NORM_3D * (dn0 + dn1 + dn2 + dn3);
    gradient = grad3.xy;
    return    TS_SIMPLEX_NORM_3D * n;
}

float ts_simplex_tile_seeded(vec2 x, ivec2 period, float alpha, uint seed,
                             out vec2 gradient) {
    uvec3 h = ts_pcg3d(uvec3(seed, seed ^ 0x9E3779B9u, seed ^ 0x85EBCA6Bu));
    float seed_alpha = float(h.x & 0xFFFFu) * (1.0 / 65536.0) * 6.28318530718; // [0, 2π)
    return ts_simplex_tile(x, period, alpha + seed_alpha, gradient);
}

struct TSWorley {
    float f1;        // nearest distance, [0, 1]
    float f2;        // second-nearest, [0, 1]
    float cell_id;   // hash of nearest cell's coordinates, [0, 1]
};

TSWorley ts_worley_tile(vec2 p, int per, float jitter, uint seed) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    float f1 = 8.0;
    float f2 = 8.0;
    float id = 0.0;

    jitter = clamp(jitter, 0.0, 1.0);

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            ivec2 neighbor = pi + ivec2(i, j);
            ivec2 wrapped  = ts_wrap2(neighbor, per);

            vec2 feature = ts_hash2_vec2(wrapped, seed);      // one call, x & y
            feature = 0.5 + (feature - 0.5) * jitter;          // shrink toward cell center

            vec2 diff = vec2(i, j) + feature - pf;
            float d2 = dot(diff, diff);

            if (d2 < f1) {
                f2 = f1;
                f1 = d2;
                id = ts_hash2(wrapped, seed);  // cell identity, already unit range
            } else if (d2 < f2) {
                f2 = d2;
            }
        }
    }

    // Normalize: max possible squared distance to nearest feature in 3×3 search
    // with jitter=1 is bounded by 2.0.  sqrt then divide by sqrt(2) → [0,1].
    const float INV_SQRT2 = 0.70710678;
    TSWorley o;
    o.f1      = clamp(sqrt(f1) * INV_SQRT2, 0.0, 1.0);
    o.f2      = clamp(sqrt(f2) * INV_SQRT2, 0.0, 1.0);
    o.cell_id = id;
    return o;
}

float ts_white_tile(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    return ts_hash2(c, seed) * 2.0 - 1.0; // [-1, 1]
}

float ts_white_pcg(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return float(h.x) * (1.0 / 4294967295.0);
}

float ts_gabor_tile(vec2 p, int per, float freq, float bandwidth,
                    float anisotropy, float angle, uint seed) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    // Remap UI bandwidth [1..16] to Gaussian sigma. Higher UI bandwidth →
    // tighter kernel. We tie sigma to 'freq' so high-frequency kernels
    // get correspondingly tighter envelopes (Gabor uncertainty principle).
    float sigma   = 1.0 / max(bandwidth, 0.5);
    float two_s2  = 2.0 * sigma * sigma;
    float r2_max  = (2.5 * sigma) * (2.5 * sigma);  // hard cutoff at 2.5σ

    float sum = 0.0;
    anisotropy = clamp(anisotropy, 0.0, 1.0);

    // 5×5 search guarantees full envelope coverage and seamless tiling.
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            ivec2 cell = ts_wrap2(pi + ivec2(i, j), per);

            // One pcg3d call → 3 decorrelated channels (kernel x, kernel y, angle)
            // instead of three separate ts_hash2 calls with seed offsets.
            vec3 h = ts_hash2_vec3(cell, seed);

            // Kernel position INSIDE the cell, [0,1)
            vec2 kpos = h.xy;
            // Vector from kernel center to sample point
            vec2 diff = vec2(i, j) + kpos - pf;
            float r2  = dot(diff, diff);
            if (r2 > r2_max) continue;

            // Anisotropic Gabor orientation
            float rand_angle = h.z * 6.28318530718;   // [0, 2π)
            float k_angle    = mix(rand_angle, angle, anisotropy);
            vec2  k_dir      = vec2(cos(k_angle), sin(k_angle));

            // Gaussian envelope × cosine carrier
            float env  = exp(-r2 / two_s2);
            float wave = cos(6.2831853 * freq * dot(diff, k_dir));
            sum += env * wave;
        }
    }

    // Empirical normalization: ~25 kernels × peak env ≈ 1.5 in worst case.
    return clamp(sum * 0.4, -1.0, 1.0);
}


// =============================================================================
// §11  FBM WRAPPERS
// =============================================================================
const float TS_OCTAVE_SEED_PRIME = 6791.0;
const uint  TS_OCTAVE_SEED_PRIME_U = 6791u;
const float TS_PI_OVER_PHI = 1.94161103873; // π / φ

// ---- Value FBM (period scales with lacunarity) ----
float ts_fbm_value(vec2 p, int period, float octaves,
                   float lacunarity, float gain, uint seed,
                   out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    vec2 gr;
    float base = ts_value_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_value_tile(p * freq, iper, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_value(vec2 p, int period, float octaves,
                   float lacunarity, float gain, uint seed) {
    vec2 g; return ts_fbm_value(p, period, octaves, lacunarity, gain, seed, g);
}

// ---- Perlin FBM ----
float ts_fbm_perlin(vec2 p, int period, float octaves,
                    float lacunarity, float gain, uint seed,
                    out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    vec2 gr;
    float base = ts_perlin_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_perlin_tile(p * freq, iper, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_perlin(vec2 p, int period, float octaves,
                    float lacunarity, float gain, uint seed) {
    vec2 g; return ts_fbm_perlin(p, period, octaves, lacunarity, gain, seed, g);
}

// ---- Simplex FBM (period scales uniformly — Y-period even-requirement
// satisfied by caller via ts_period_even) ----
float ts_fbm_simplex(vec2 p, ivec2 period, float octaves,
                     float lacunarity, float gain,
                     float alpha, uint seed,
                     out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int lac_int = int(lacunarity);
    ivec2 per_i = ivec2(
        max(period.x, 1),
        max(period.y, 1)
    );
    uint os = seed;
    float oct_alpha = alpha;
    vec2 gr;
    float base = ts_simplex_tile_seeded(p * freq, per_i, oct_alpha, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        per_i *= lac_int;
        oct_alpha = alpha + oct * TS_PI_OVER_PHI;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_simplex_tile_seeded(p * freq, per_i, oct_alpha, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_simplex(vec2 p, ivec2 period, float octaves,
                     float lacunarity, float gain,
                     float alpha, uint seed) {
    vec2 g;
    return ts_fbm_simplex(p, period, octaves, lacunarity, gain, alpha, seed, g);
}


TSWorley ts_fbm_worley(vec2 p, int period, float octaves,
                       float lacunarity, float gain, float jitter, uint seed) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    TSWorley base = ts_worley_tile(p * freq, iper, jitter, os);
    float f1_sum = base.f1, f2_sum = base.f2, id_sum = base.cell_id;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) {
        TSWorley o;
        o.f1 = clamp(f1_sum, 0.0, 1.0);
        o.f2 = clamp(f2_sum, 0.0, 1.0);
        o.cell_id = clamp(id_sum, 0.0, 1.0);
        return o;
    }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        TSWorley w2 = ts_worley_tile(p * freq, iper, jitter, os);
        f1_sum += weight * w2.f1;
        f2_sum += weight * w2.f2;
        id_sum += weight * w2.cell_id;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    TSWorley o;
    o.f1 = clamp(f1_sum * inv, 0.0, 1.0);
    o.f2 = clamp(f2_sum * inv, 0.0, 1.0);
    o.cell_id = clamp(id_sum * inv, 0.0, 1.0);
    return o;
}

// ---- Gabor FBM ----
float ts_fbm_gabor(vec2 p, int period, float octaves,
                   float lacunarity, float gain,
                   float freq_hz, float bandwidth,
                   float anisotropy, float angle, uint seed) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    float base = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth, anisotropy, angle, os);
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return clamp(base, -1.0, 1.0); }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        float n = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth, anisotropy, angle, os);
        base += weight * n;
        norm += weight;
    } while (oct < octaves);
    return clamp(base / max(norm, 1e-6), -1.0, 1.0);
}

// =============================================================================
// REMAP HELPERS
// =============================================================================
// All FBMs return [-1, 1].  want [0, 1].
float ts_to_unit(float n)  { return n * 0.5 + 0.5; }
vec2  ts_to_unit(vec2  n)  { return n * 0.5 + 0.5; }

#endif // TS_NOISE_COMMON

vec4 node_worley(vec2 uv,
                 float period,
                 float octaves,
                 float lacunarity,
                 float roughness,
                 float jitter,
                 float speed,
                 float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);
    float jit = clamp(jitter, 0.0, 1.0);

    vec2 p = fract(uv) * float(iper)
           + vec2(pc.time * speed);

    float r = ts_fbm_worley(p, iper, octaves, lacunarity, roughness, jit, iseed).f1;
    float g = ts_fbm_worley(p, iper, octaves, lacunarity, roughness, jit, iseed + 379u).f1;
    float b = ts_fbm_worley(p, iper, octaves, lacunarity, roughness, jit, iseed + 757u).f1;
    float a = ts_fbm_worley(p, iper, octaves, lacunarity, roughness, jit, iseed + 1013u).f1;

    return vec4(r, g, b, a);
}



#ifndef TS_REMAP
#define TS_REMAP

float ts_remap(float v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return out_min;
    return out_min + (out_max - out_min) * (v - in_min) / denom;
}

vec2 ts_remap(vec2 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec2(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec3 ts_remap(vec3 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec3(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec4 ts_remap(vec4 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec4(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

#endif

#ifndef TS_NOISE_COMMON
#define TS_NOISE_COMMON

//PCG INTEGER HASH
uvec2 ts_pcg2d(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    return v;
}

uvec3 ts_pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

uvec4 ts_pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    uint x = v.x + v.y * v.w;
    uint y = v.y + v.z * x;
    uint z = v.z + v.x * y;
    uint w = v.w + y * z;
    v = uvec4(x, y, z, w) ^ (uvec4(x, y, z, w) >> 16u);
    x = v.x + v.y * v.w;
    y = v.y + v.z * x;
    z = v.z + v.x * y;
    w = v.w + y * z;
    return uvec4(x, y, z, w);
}



// LATTICE HASHES  (pcg3d-based — seed rides natively in .z)
const float TS_HASH_TO_ANGLE = 21.548;

float ts_hash2(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return float(h.x & 0xFFFFu) * (1.0 / 65536.0);
}

vec4 ts_hash2_quad(ivec2 c00, uint seed) {
    uvec3 h00 = ts_pcg3d(uvec3(uvec2(c00),                seed));
    uvec3 h10 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(1u, 0u), seed));
    uvec3 h01 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(0u, 1u), seed));
    uvec3 h11 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(1u, 1u), seed));
    return vec4(h00.x & 0xFFFFu, h10.x & 0xFFFFu, h01.x & 0xFFFFu, h11.x & 0xFFFFu)
         * (1.0 / 65536.0);
}

vec2 ts_hash2_vec2(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return vec2(h.x & 0xFFFFu, h.y & 0xFFFFu) * (1.0 / 65536.0);
}

vec3 ts_hash2_vec3(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return vec3(h.x & 0xFFFFu, h.y & 0xFFFFu, h.z & 0xFFFFu) * (1.0 / 65536.0);
}

vec2 ts_grad8(float hash01) {
    float a = floor(hash01 * 8.0) * 0.78539816339; // π/4
    return vec2(cos(a), sin(a));
}

vec2 ts_grad16(float hash01) {
    float a = floor(hash01 * 16.0) * 0.39269908170; // π/8
    return vec2(cos(a), sin(a));
}

vec2 ts_grad_continuous(float hash01) {
    float a = hash01 * TS_HASH_TO_ANGLE;
    return vec2(cos(a), sin(a));
}


// SMOOTHSTEP
float ts_quintic(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

vec2 ts_quintic(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float ts_quintic_d(float t) {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

vec2 ts_quintic_d(vec2 t) {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}


// PERIOD
int ts_period_int(float p) {
    return max(int(round(p)), 1);
}

int ts_period_pow2(float p) {
    int v = max(int(round(p)), 1);
    float l = floor(log2(float(v)));
    int lo = int(l);
    int pow2_lo = 1 << lo;
    int pow2_hi = pow2_lo * 2;
    return (v - pow2_lo <= pow2_hi - v) ? pow2_lo : pow2_hi;
}

int ts_period_even(float p) {
    return max(int(round(p * 0.5)) * 2, 2);
}


int ts_wrap(int v, int per) {
    return int(mod(float(v), float(per)));
}

ivec2 ts_wrap2(ivec2 v, int per) {
    return ivec2(ts_wrap(v.x, per), ts_wrap(v.y, per));
}


float ts_value_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,               per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0), per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1), per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1), per);

    // Corner values: unit hash → [-1, 1]
    float v00 = ts_hash2(c00, seed) * 2.0 - 1.0;
    float v10 = ts_hash2(c10, seed) * 2.0 - 1.0;
    float v01 = ts_hash2(c01, seed) * 2.0 - 1.0;
    float v11 = ts_hash2(c11, seed) * 2.0 - 1.0;

    vec2 u  = ts_quintic(pf);
    vec2 du = ts_quintic_d(pf);

    float a = mix(v00, v10, u.x);
    float b = mix(v01, v11, u.x);
    float n = mix(a, b, u.y);

    // ∂n/∂x = du.x × [(v10-v00)(1-u.y) + (v11-v01)u.y]
    // ∂n/∂y = du.y × [(v01-v00)(1-u.x) + (v11-v10)u.x]
    grad.x = du.x * mix(v10 - v00, v11 - v01, u.y);
    grad.y = du.y * mix(v01 - v00, v11 - v10, u.x);

    return n;
}

float ts_value_tile(vec2 p, int per, uint seed) {
    vec2 g;
    return ts_value_tile(p, per, seed, g);
}

const float TS_PERLIN_NORM = 1.3393713;   // = 1.0 / 0.74682413

float ts_perlin_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,               per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0), per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1), per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1), per);

    vec2 g00 = ts_grad8(ts_hash2(c00, seed));
    vec2 g10 = ts_grad8(ts_hash2(c10, seed));
    vec2 g01 = ts_grad8(ts_hash2(c01, seed));
    vec2 g11 = ts_grad8(ts_hash2(c11, seed));

    vec2 d00 = pf;
    vec2 d10 = pf - vec2(1.0, 0.0);
    vec2 d01 = pf - vec2(0.0, 1.0);
    vec2 d11 = pf - vec2(1.0, 1.0);

    float n00 = dot(g00, d00);
    float n10 = dot(g10, d10);
    float n01 = dot(g01, d01);
    float n11 = dot(g11, d11);

    vec2 u  = ts_quintic(pf);
    vec2 du = ts_quintic_d(pf);

    float a = mix(n00, n10, u.x);
    float b = mix(n01, n11, u.x);
    float n = mix(a, b, u.y);

    // Analytical gradient (chain rule through bilinear blend + dot products)
    vec2 ga = mix(g00, g10, u.x) + du.x * (n10 - n00) * vec2(1.0, 0.0);
    vec2 gb = mix(g01, g11, u.x) + du.x * (n11 - n01) * vec2(1.0, 0.0);
    grad   = mix(ga, gb, u.y) + du.y * (b - a) * vec2(0.0, 1.0);
    grad  *= TS_PERLIN_NORM;

    return n * TS_PERLIN_NORM;
}

float ts_perlin_tile(vec2 p, int per, uint seed) {
    vec2 g;
    return ts_perlin_tile(p, per, seed, g);
}

const float TS_SIMPLEX_NORM_3D = 39.5;

// 3D axis-aligned simplex grid (tetrahedral lattice) evaluated on a 2D slice
// (z=0). Tiles seamlessly at ANY integer period in x and y.
// Based on Gustavson & McEwan, JCGT 11(1), 2022 — psrdnoise3.
float ts_simplex_tile(vec2 x, ivec2 period, float alpha, out vec2 gradient) {
    // Promote to 3D: z=0 slice, no Z wrapping
    vec3 x3 = vec3(x.x, x.y, 0.0);

    // Axis-aligned simplex grid transform: uvw = M * x
    // M = [[0,1,1],[1,0,1],[1,1,0]]
    vec3 uvw = vec3(x3.y + x3.z, x3.x + x3.z, x3.x + x3.y);

    vec3 i0 = floor(uvw);
    vec3 f0 = fract(uvw);

    // Rank-order fractional parts to determine tetrahedron corners
    vec3 g_ = step(f0.xyx, f0.yzz);
    vec3 l_ = 1.0 - g_;
    vec3 g  = vec3(l_.z, g_.xy);
    vec3 l  = vec3(l_.xy, g_.z);
    vec3 o1 = min(g, l);
    vec3 o2 = max(g, l);

    vec3 i1 = i0 + o1;
    vec3 i2 = i0 + o2;
    vec3 i3 = i0 + vec3(1.0);

    // Inverse transform Mi = [[-0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5]]
    vec3 v0 = vec3(-0.5*i0.x + 0.5*i0.y + 0.5*i0.z,
                     0.5*i0.x - 0.5*i0.y + 0.5*i0.z,
                     0.5*i0.x + 0.5*i0.y - 0.5*i0.z);
    vec3 v1 = vec3(-0.5*i1.x + 0.5*i1.y + 0.5*i1.z,
                     0.5*i1.x - 0.5*i1.y + 0.5*i1.z,
                     0.5*i1.x + 0.5*i1.y - 0.5*i1.z);
    vec3 v2 = vec3(-0.5*i2.x + 0.5*i2.y + 0.5*i2.z,
                     0.5*i2.x - 0.5*i2.y + 0.5*i2.z,
                     0.5*i2.x + 0.5*i2.y - 0.5*i2.z);
    vec3 v3 = vec3(-0.5*i3.x + 0.5*i3.y + 0.5*i3.z,
                     0.5*i3.x - 0.5*i3.y + 0.5*i3.z,
                     0.5*i3.x + 0.5*i3.y - 0.5*i3.z);

    // Displacement vectors (in texture space)
    vec3 x0 = x3 - v0;
    vec3 x1 = x3 - v1;
    vec3 x2 = x3 - v2;
    vec3 x3_ = x3 - v3;

    // Wrap corner positions and transform back to simplex space
    if (period.x > 0 || period.y > 0) {
        vec4 vx = vec4(v0.x, v1.x, v2.x, v3.x);
        vec4 vy = vec4(v0.y, v1.y, v2.y, v3.y);
        vec4 vz = vec4(v0.z, v1.z, v2.z, v3.z);

        if (period.x > 0) vx = mod(vx, float(period.x));
        if (period.y > 0) vy = mod(vy, float(period.y));

        // Transform wrapped texture-space positions back to simplex space
        // M = [[0,1,1],[1,0,1],[1,1,0]]: M*v = (v.y+v.z, v.x+v.z, v.x+v.y)
        vec3 w0 = vec3(vy.x + vz.x, vx.x + vz.x, vx.x + vy.x);
        vec3 w1 = vec3(vy.y + vz.y, vx.y + vz.y, vx.y + vy.y);
        vec3 w2 = vec3(vy.z + vz.z, vx.z + vz.z, vx.z + vy.z);
        vec3 w3 = vec3(vy.w + vz.w, vx.w + vz.w, vx.w + vy.w);

        // Fix rounding errors
        i0 = floor(w0 + 0.5);
        i1 = floor(w1 + 0.5);
        i2 = floor(w2 + 0.5);
        i3 = floor(w3 + 0.5);
    }

    // Hash each corner via pcg3d
    uvec3 h0 = ts_pcg3d(uvec3(uint(i0.x), uint(i0.y), uint(i0.z)));
    uvec3 h1 = ts_pcg3d(uvec3(uint(i1.x), uint(i1.y), uint(i1.z)));
    uvec3 h2 = ts_pcg3d(uvec3(uint(i2.x), uint(i2.y), uint(i2.z)));
    uvec3 h3 = ts_pcg3d(uvec3(uint(i3.x), uint(i3.y), uint(i3.z)));

    // Generate gradient direction from hash
    float hf0 = float(h0.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf1 = float(h1.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf2 = float(h2.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf3 = float(h3.x & 0xFFFFu) * (1.0 / 65536.0);

    vec4 psi = vec4(hf0, hf1, hf2, hf3) * TS_HASH_TO_ANGLE + alpha;
    // 2D gradients (xy only — z=0 slice)
    vec4 gx = cos(psi);
    vec4 gy = sin(psi);

    vec3 g0 = vec3(gx.x, gy.x, 0.0);
    vec3 g1 = vec3(gx.y, gy.y, 0.0);
    vec3 g2 = vec3(gx.z, gy.z, 0.0);
    vec3 g3 = vec3(gx.w, gy.w, 0.0);

    // Radial decay kernel (3D: 0.5 support radius)
    vec4 w  = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3_,x3_)), 0.0);
    vec4 w2 = w * w;
    vec4 w3 = w2 * w;

    vec4 gdotx = vec4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3_));
    float n = dot(w3, gdotx);

    // Analytical gradient
    vec4 dw  = -6.0 * w2 * gdotx;
    vec3 dn0 = w3.x * g0 + dw.x * x0;
    vec3 dn1 = w3.y * g1 + dw.y * x1;
    vec3 dn2 = w3.z * g2 + dw.z * x2;
    vec3 dn3 = w3.w * g3 + dw.w * x3_;

    vec3 grad3 = TS_SIMPLEX_NORM_3D * (dn0 + dn1 + dn2 + dn3);
    gradient = grad3.xy;
    return    TS_SIMPLEX_NORM_3D * n;
}

float ts_simplex_tile_seeded(vec2 x, ivec2 period, float alpha, uint seed,
                             out vec2 gradient) {
    uvec3 h = ts_pcg3d(uvec3(seed, seed ^ 0x9E3779B9u, seed ^ 0x85EBCA6Bu));
    float seed_alpha = float(h.x & 0xFFFFu) * (1.0 / 65536.0) * 6.28318530718; // [0, 2π)
    return ts_simplex_tile(x, period, alpha + seed_alpha, gradient);
}

struct TSWorley {
    float f1;        // nearest distance, [0, 1]
    float f2;        // second-nearest, [0, 1]
    float cell_id;   // hash of nearest cell's coordinates, [0, 1]
};

TSWorley ts_worley_tile(vec2 p, int per, float jitter, uint seed) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    float f1 = 8.0;
    float f2 = 8.0;
    float id = 0.0;

    jitter = clamp(jitter, 0.0, 1.0);

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            ivec2 neighbor = pi + ivec2(i, j);
            ivec2 wrapped  = ts_wrap2(neighbor, per);

            vec2 feature = ts_hash2_vec2(wrapped, seed);      // one call, x & y
            feature = 0.5 + (feature - 0.5) * jitter;          // shrink toward cell center

            vec2 diff = vec2(i, j) + feature - pf;
            float d2 = dot(diff, diff);

            if (d2 < f1) {
                f2 = f1;
                f1 = d2;
                id = ts_hash2(wrapped, seed);  // cell identity, already unit range
            } else if (d2 < f2) {
                f2 = d2;
            }
        }
    }

    // Normalize: max possible squared distance to nearest feature in 3×3 search
    // with jitter=1 is bounded by 2.0.  sqrt then divide by sqrt(2) → [0,1].
    const float INV_SQRT2 = 0.70710678;
    TSWorley o;
    o.f1      = clamp(sqrt(f1) * INV_SQRT2, 0.0, 1.0);
    o.f2      = clamp(sqrt(f2) * INV_SQRT2, 0.0, 1.0);
    o.cell_id = id;
    return o;
}

float ts_white_tile(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    return ts_hash2(c, seed) * 2.0 - 1.0; // [-1, 1]
}

float ts_white_pcg(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return float(h.x) * (1.0 / 4294967295.0);
}

float ts_gabor_tile(vec2 p, int per, float freq, float bandwidth,
                    float anisotropy, float angle, uint seed) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    // Remap UI bandwidth [1..16] to Gaussian sigma. Higher UI bandwidth →
    // tighter kernel. We tie sigma to 'freq' so high-frequency kernels
    // get correspondingly tighter envelopes (Gabor uncertainty principle).
    float sigma   = 1.0 / max(bandwidth, 0.5);
    float two_s2  = 2.0 * sigma * sigma;
    float r2_max  = (2.5 * sigma) * (2.5 * sigma);  // hard cutoff at 2.5σ

    float sum = 0.0;
    anisotropy = clamp(anisotropy, 0.0, 1.0);

    // 5×5 search guarantees full envelope coverage and seamless tiling.
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            ivec2 cell = ts_wrap2(pi + ivec2(i, j), per);

            // One pcg3d call → 3 decorrelated channels (kernel x, kernel y, angle)
            // instead of three separate ts_hash2 calls with seed offsets.
            vec3 h = ts_hash2_vec3(cell, seed);

            // Kernel position INSIDE the cell, [0,1)
            vec2 kpos = h.xy;
            // Vector from kernel center to sample point
            vec2 diff = vec2(i, j) + kpos - pf;
            float r2  = dot(diff, diff);
            if (r2 > r2_max) continue;

            // Anisotropic Gabor orientation
            float rand_angle = h.z * 6.28318530718;   // [0, 2π)
            float k_angle    = mix(rand_angle, angle, anisotropy);
            vec2  k_dir      = vec2(cos(k_angle), sin(k_angle));

            // Gaussian envelope × cosine carrier
            float env  = exp(-r2 / two_s2);
            float wave = cos(6.2831853 * freq * dot(diff, k_dir));
            sum += env * wave;
        }
    }

    // Empirical normalization: ~25 kernels × peak env ≈ 1.5 in worst case.
    return clamp(sum * 0.4, -1.0, 1.0);
}


// =============================================================================
// §11  FBM WRAPPERS
// =============================================================================
const float TS_OCTAVE_SEED_PRIME = 6791.0;
const uint  TS_OCTAVE_SEED_PRIME_U = 6791u;
const float TS_PI_OVER_PHI = 1.94161103873; // π / φ

// ---- Value FBM (period scales with lacunarity) ----
float ts_fbm_value(vec2 p, int period, float octaves,
                   float lacunarity, float gain, uint seed,
                   out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    vec2 gr;
    float base = ts_value_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_value_tile(p * freq, iper, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_value(vec2 p, int period, float octaves,
                   float lacunarity, float gain, uint seed) {
    vec2 g; return ts_fbm_value(p, period, octaves, lacunarity, gain, seed, g);
}

// ---- Perlin FBM ----
float ts_fbm_perlin(vec2 p, int period, float octaves,
                    float lacunarity, float gain, uint seed,
                    out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    vec2 gr;
    float base = ts_perlin_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_perlin_tile(p * freq, iper, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_perlin(vec2 p, int period, float octaves,
                    float lacunarity, float gain, uint seed) {
    vec2 g; return ts_fbm_perlin(p, period, octaves, lacunarity, gain, seed, g);
}

// ---- Simplex FBM (period scales uniformly — Y-period even-requirement
// satisfied by caller via ts_period_even) ----
float ts_fbm_simplex(vec2 p, ivec2 period, float octaves,
                     float lacunarity, float gain,
                     float alpha, uint seed,
                     out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int lac_int = int(lacunarity);
    ivec2 per_i = ivec2(
        max(period.x, 1),
        max(period.y, 1)
    );
    uint os = seed;
    float oct_alpha = alpha;
    vec2 gr;
    float base = ts_simplex_tile_seeded(p * freq, per_i, oct_alpha, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        per_i *= lac_int;
        oct_alpha = alpha + oct * TS_PI_OVER_PHI;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_simplex_tile_seeded(p * freq, per_i, oct_alpha, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_simplex(vec2 p, ivec2 period, float octaves,
                     float lacunarity, float gain,
                     float alpha, uint seed) {
    vec2 g;
    return ts_fbm_simplex(p, period, octaves, lacunarity, gain, alpha, seed, g);
}


TSWorley ts_fbm_worley(vec2 p, int period, float octaves,
                       float lacunarity, float gain, float jitter, uint seed) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    TSWorley base = ts_worley_tile(p * freq, iper, jitter, os);
    float f1_sum = base.f1, f2_sum = base.f2, id_sum = base.cell_id;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) {
        TSWorley o;
        o.f1 = clamp(f1_sum, 0.0, 1.0);
        o.f2 = clamp(f2_sum, 0.0, 1.0);
        o.cell_id = clamp(id_sum, 0.0, 1.0);
        return o;
    }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        TSWorley w2 = ts_worley_tile(p * freq, iper, jitter, os);
        f1_sum += weight * w2.f1;
        f2_sum += weight * w2.f2;
        id_sum += weight * w2.cell_id;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    TSWorley o;
    o.f1 = clamp(f1_sum * inv, 0.0, 1.0);
    o.f2 = clamp(f2_sum * inv, 0.0, 1.0);
    o.cell_id = clamp(id_sum * inv, 0.0, 1.0);
    return o;
}

// ---- Gabor FBM ----
float ts_fbm_gabor(vec2 p, int period, float octaves,
                   float lacunarity, float gain,
                   float freq_hz, float bandwidth,
                   float anisotropy, float angle, uint seed) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    float base = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth, anisotropy, angle, os);
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return clamp(base, -1.0, 1.0); }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        float n = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth, anisotropy, angle, os);
        base += weight * n;
        norm += weight;
    } while (oct < octaves);
    return clamp(base / max(norm, 1e-6), -1.0, 1.0);
}

// =============================================================================
// REMAP HELPERS
// =============================================================================
// All FBMs return [-1, 1].  want [0, 1].
float ts_to_unit(float n)  { return n * 0.5 + 0.5; }
vec2  ts_to_unit(vec2  n)  { return n * 0.5 + 0.5; }

#endif // TS_NOISE_COMMON

vec4 node_simplex(vec2 uv,
                  float period,
                  float octaves,
                  float lacunarity,
                  float roughness,
                  float speed,
                  float rotation,
                  float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);

    vec2 p = uv * float(iper)
           + vec2(pc.time * speed);

    ivec2 per = ivec2(iper);

    float r = ts_fbm_simplex(p, per, octaves, lacunarity, roughness, rotation, iseed);
    float g = ts_fbm_simplex(p, per, octaves, lacunarity, roughness, rotation, iseed + 379u);
    float b = ts_fbm_simplex(p, per, octaves, lacunarity, roughness, rotation, iseed + 757u);
    float a = ts_fbm_simplex(p, per, octaves, lacunarity, roughness, rotation, iseed + 1013u);

    return vec4(ts_to_unit(r), ts_to_unit(g), ts_to_unit(b), ts_to_unit(a));
}



#ifndef TS_REMAP
#define TS_REMAP

float ts_remap(float v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return out_min;
    return out_min + (out_max - out_min) * (v - in_min) / denom;
}

vec2 ts_remap(vec2 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec2(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec3 ts_remap(vec3 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec3(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

vec4 ts_remap(vec4 v, float in_min, float in_max, float out_min, float out_max) {
    float denom = in_max - in_min;
    if (abs(denom) < 1e-8) return vec4(out_min);
    float scale = (out_max - out_min) / denom;
    return out_min + (v - in_min) * scale;
}

#endif

#ifndef TS_NOISE_COMMON
#define TS_NOISE_COMMON

//PCG INTEGER HASH
uvec2 ts_pcg2d(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    return v;
}

uvec3 ts_pcg3d(uvec3 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

uvec4 ts_pcg4d(uvec4 v) {
    v = v * 1664525u + 1013904223u;
    uint x = v.x + v.y * v.w;
    uint y = v.y + v.z * x;
    uint z = v.z + v.x * y;
    uint w = v.w + y * z;
    v = uvec4(x, y, z, w) ^ (uvec4(x, y, z, w) >> 16u);
    x = v.x + v.y * v.w;
    y = v.y + v.z * x;
    z = v.z + v.x * y;
    w = v.w + y * z;
    return uvec4(x, y, z, w);
}



// LATTICE HASHES  (pcg3d-based — seed rides natively in .z)
const float TS_HASH_TO_ANGLE = 21.548;

float ts_hash2(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return float(h.x & 0xFFFFu) * (1.0 / 65536.0);
}

vec4 ts_hash2_quad(ivec2 c00, uint seed) {
    uvec3 h00 = ts_pcg3d(uvec3(uvec2(c00),                seed));
    uvec3 h10 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(1u, 0u), seed));
    uvec3 h01 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(0u, 1u), seed));
    uvec3 h11 = ts_pcg3d(uvec3(uvec2(c00) + uvec2(1u, 1u), seed));
    return vec4(h00.x & 0xFFFFu, h10.x & 0xFFFFu, h01.x & 0xFFFFu, h11.x & 0xFFFFu)
         * (1.0 / 65536.0);
}

vec2 ts_hash2_vec2(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return vec2(h.x & 0xFFFFu, h.y & 0xFFFFu) * (1.0 / 65536.0);
}

vec3 ts_hash2_vec3(ivec2 c, uint seed) {
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return vec3(h.x & 0xFFFFu, h.y & 0xFFFFu, h.z & 0xFFFFu) * (1.0 / 65536.0);
}

vec2 ts_grad8(float hash01) {
    float a = floor(hash01 * 8.0) * 0.78539816339; // π/4
    return vec2(cos(a), sin(a));
}

vec2 ts_grad16(float hash01) {
    float a = floor(hash01 * 16.0) * 0.39269908170; // π/8
    return vec2(cos(a), sin(a));
}

vec2 ts_grad_continuous(float hash01) {
    float a = hash01 * TS_HASH_TO_ANGLE;
    return vec2(cos(a), sin(a));
}


// SMOOTHSTEP
float ts_quintic(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

vec2 ts_quintic(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float ts_quintic_d(float t) {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

vec2 ts_quintic_d(vec2 t) {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}


// PERIOD
int ts_period_int(float p) {
    return max(int(round(p)), 1);
}

int ts_period_pow2(float p) {
    int v = max(int(round(p)), 1);
    float l = floor(log2(float(v)));
    int lo = int(l);
    int pow2_lo = 1 << lo;
    int pow2_hi = pow2_lo * 2;
    return (v - pow2_lo <= pow2_hi - v) ? pow2_lo : pow2_hi;
}

int ts_period_even(float p) {
    return max(int(round(p * 0.5)) * 2, 2);
}


int ts_wrap(int v, int per) {
    return int(mod(float(v), float(per)));
}

ivec2 ts_wrap2(ivec2 v, int per) {
    return ivec2(ts_wrap(v.x, per), ts_wrap(v.y, per));
}


float ts_value_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,               per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0), per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1), per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1), per);

    // Corner values: unit hash → [-1, 1]
    float v00 = ts_hash2(c00, seed) * 2.0 - 1.0;
    float v10 = ts_hash2(c10, seed) * 2.0 - 1.0;
    float v01 = ts_hash2(c01, seed) * 2.0 - 1.0;
    float v11 = ts_hash2(c11, seed) * 2.0 - 1.0;

    vec2 u  = ts_quintic(pf);
    vec2 du = ts_quintic_d(pf);

    float a = mix(v00, v10, u.x);
    float b = mix(v01, v11, u.x);
    float n = mix(a, b, u.y);

    // ∂n/∂x = du.x × [(v10-v00)(1-u.y) + (v11-v01)u.y]
    // ∂n/∂y = du.y × [(v01-v00)(1-u.x) + (v11-v10)u.x]
    grad.x = du.x * mix(v10 - v00, v11 - v01, u.y);
    grad.y = du.y * mix(v01 - v00, v11 - v10, u.x);

    return n;
}

float ts_value_tile(vec2 p, int per, uint seed) {
    vec2 g;
    return ts_value_tile(p, per, seed, g);
}

const float TS_PERLIN_NORM = 1.3393713;   // = 1.0 / 0.74682413

float ts_perlin_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,               per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0), per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1), per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1), per);

    vec2 g00 = ts_grad8(ts_hash2(c00, seed));
    vec2 g10 = ts_grad8(ts_hash2(c10, seed));
    vec2 g01 = ts_grad8(ts_hash2(c01, seed));
    vec2 g11 = ts_grad8(ts_hash2(c11, seed));

    vec2 d00 = pf;
    vec2 d10 = pf - vec2(1.0, 0.0);
    vec2 d01 = pf - vec2(0.0, 1.0);
    vec2 d11 = pf - vec2(1.0, 1.0);

    float n00 = dot(g00, d00);
    float n10 = dot(g10, d10);
    float n01 = dot(g01, d01);
    float n11 = dot(g11, d11);

    vec2 u  = ts_quintic(pf);
    vec2 du = ts_quintic_d(pf);

    float a = mix(n00, n10, u.x);
    float b = mix(n01, n11, u.x);
    float n = mix(a, b, u.y);

    // Analytical gradient (chain rule through bilinear blend + dot products)
    vec2 ga = mix(g00, g10, u.x) + du.x * (n10 - n00) * vec2(1.0, 0.0);
    vec2 gb = mix(g01, g11, u.x) + du.x * (n11 - n01) * vec2(1.0, 0.0);
    grad   = mix(ga, gb, u.y) + du.y * (b - a) * vec2(0.0, 1.0);
    grad  *= TS_PERLIN_NORM;

    return n * TS_PERLIN_NORM;
}

float ts_perlin_tile(vec2 p, int per, uint seed) {
    vec2 g;
    return ts_perlin_tile(p, per, seed, g);
}

const float TS_SIMPLEX_NORM_3D = 39.5;

// 3D axis-aligned simplex grid (tetrahedral lattice) evaluated on a 2D slice
// (z=0). Tiles seamlessly at ANY integer period in x and y.
// Based on Gustavson & McEwan, JCGT 11(1), 2022 — psrdnoise3.
float ts_simplex_tile(vec2 x, ivec2 period, float alpha, out vec2 gradient) {
    // Promote to 3D: z=0 slice, no Z wrapping
    vec3 x3 = vec3(x.x, x.y, 0.0);

    // Axis-aligned simplex grid transform: uvw = M * x
    // M = [[0,1,1],[1,0,1],[1,1,0]]
    vec3 uvw = vec3(x3.y + x3.z, x3.x + x3.z, x3.x + x3.y);

    vec3 i0 = floor(uvw);
    vec3 f0 = fract(uvw);

    // Rank-order fractional parts to determine tetrahedron corners
    vec3 g_ = step(f0.xyx, f0.yzz);
    vec3 l_ = 1.0 - g_;
    vec3 g  = vec3(l_.z, g_.xy);
    vec3 l  = vec3(l_.xy, g_.z);
    vec3 o1 = min(g, l);
    vec3 o2 = max(g, l);

    vec3 i1 = i0 + o1;
    vec3 i2 = i0 + o2;
    vec3 i3 = i0 + vec3(1.0);

    // Inverse transform Mi = [[-0.5,0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,-0.5]]
    vec3 v0 = vec3(-0.5*i0.x + 0.5*i0.y + 0.5*i0.z,
                     0.5*i0.x - 0.5*i0.y + 0.5*i0.z,
                     0.5*i0.x + 0.5*i0.y - 0.5*i0.z);
    vec3 v1 = vec3(-0.5*i1.x + 0.5*i1.y + 0.5*i1.z,
                     0.5*i1.x - 0.5*i1.y + 0.5*i1.z,
                     0.5*i1.x + 0.5*i1.y - 0.5*i1.z);
    vec3 v2 = vec3(-0.5*i2.x + 0.5*i2.y + 0.5*i2.z,
                     0.5*i2.x - 0.5*i2.y + 0.5*i2.z,
                     0.5*i2.x + 0.5*i2.y - 0.5*i2.z);
    vec3 v3 = vec3(-0.5*i3.x + 0.5*i3.y + 0.5*i3.z,
                     0.5*i3.x - 0.5*i3.y + 0.5*i3.z,
                     0.5*i3.x + 0.5*i3.y - 0.5*i3.z);

    // Displacement vectors (in texture space)
    vec3 x0 = x3 - v0;
    vec3 x1 = x3 - v1;
    vec3 x2 = x3 - v2;
    vec3 x3_ = x3 - v3;

    // Wrap corner positions and transform back to simplex space
    if (period.x > 0 || period.y > 0) {
        vec4 vx = vec4(v0.x, v1.x, v2.x, v3.x);
        vec4 vy = vec4(v0.y, v1.y, v2.y, v3.y);
        vec4 vz = vec4(v0.z, v1.z, v2.z, v3.z);

        if (period.x > 0) vx = mod(vx, float(period.x));
        if (period.y > 0) vy = mod(vy, float(period.y));

        // Transform wrapped texture-space positions back to simplex space
        // M = [[0,1,1],[1,0,1],[1,1,0]]: M*v = (v.y+v.z, v.x+v.z, v.x+v.y)
        vec3 w0 = vec3(vy.x + vz.x, vx.x + vz.x, vx.x + vy.x);
        vec3 w1 = vec3(vy.y + vz.y, vx.y + vz.y, vx.y + vy.y);
        vec3 w2 = vec3(vy.z + vz.z, vx.z + vz.z, vx.z + vy.z);
        vec3 w3 = vec3(vy.w + vz.w, vx.w + vz.w, vx.w + vy.w);

        // Fix rounding errors
        i0 = floor(w0 + 0.5);
        i1 = floor(w1 + 0.5);
        i2 = floor(w2 + 0.5);
        i3 = floor(w3 + 0.5);
    }

    // Hash each corner via pcg3d
    uvec3 h0 = ts_pcg3d(uvec3(uint(i0.x), uint(i0.y), uint(i0.z)));
    uvec3 h1 = ts_pcg3d(uvec3(uint(i1.x), uint(i1.y), uint(i1.z)));
    uvec3 h2 = ts_pcg3d(uvec3(uint(i2.x), uint(i2.y), uint(i2.z)));
    uvec3 h3 = ts_pcg3d(uvec3(uint(i3.x), uint(i3.y), uint(i3.z)));

    // Generate gradient direction from hash
    float hf0 = float(h0.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf1 = float(h1.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf2 = float(h2.x & 0xFFFFu) * (1.0 / 65536.0);
    float hf3 = float(h3.x & 0xFFFFu) * (1.0 / 65536.0);

    vec4 psi = vec4(hf0, hf1, hf2, hf3) * TS_HASH_TO_ANGLE + alpha;
    // 2D gradients (xy only — z=0 slice)
    vec4 gx = cos(psi);
    vec4 gy = sin(psi);

    vec3 g0 = vec3(gx.x, gy.x, 0.0);
    vec3 g1 = vec3(gx.y, gy.y, 0.0);
    vec3 g2 = vec3(gx.z, gy.z, 0.0);
    vec3 g3 = vec3(gx.w, gy.w, 0.0);

    // Radial decay kernel (3D: 0.5 support radius)
    vec4 w  = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3_,x3_)), 0.0);
    vec4 w2 = w * w;
    vec4 w3 = w2 * w;

    vec4 gdotx = vec4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3_));
    float n = dot(w3, gdotx);

    // Analytical gradient
    vec4 dw  = -6.0 * w2 * gdotx;
    vec3 dn0 = w3.x * g0 + dw.x * x0;
    vec3 dn1 = w3.y * g1 + dw.y * x1;
    vec3 dn2 = w3.z * g2 + dw.z * x2;
    vec3 dn3 = w3.w * g3 + dw.w * x3_;

    vec3 grad3 = TS_SIMPLEX_NORM_3D * (dn0 + dn1 + dn2 + dn3);
    gradient = grad3.xy;
    return    TS_SIMPLEX_NORM_3D * n;
}

float ts_simplex_tile_seeded(vec2 x, ivec2 period, float alpha, uint seed,
                             out vec2 gradient) {
    uvec3 h = ts_pcg3d(uvec3(seed, seed ^ 0x9E3779B9u, seed ^ 0x85EBCA6Bu));
    float seed_alpha = float(h.x & 0xFFFFu) * (1.0 / 65536.0) * 6.28318530718; // [0, 2π)
    return ts_simplex_tile(x, period, alpha + seed_alpha, gradient);
}

struct TSWorley {
    float f1;        // nearest distance, [0, 1]
    float f2;        // second-nearest, [0, 1]
    float cell_id;   // hash of nearest cell's coordinates, [0, 1]
};

TSWorley ts_worley_tile(vec2 p, int per, float jitter, uint seed) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    float f1 = 8.0;
    float f2 = 8.0;
    float id = 0.0;

    jitter = clamp(jitter, 0.0, 1.0);

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            ivec2 neighbor = pi + ivec2(i, j);
            ivec2 wrapped  = ts_wrap2(neighbor, per);

            vec2 feature = ts_hash2_vec2(wrapped, seed);      // one call, x & y
            feature = 0.5 + (feature - 0.5) * jitter;          // shrink toward cell center

            vec2 diff = vec2(i, j) + feature - pf;
            float d2 = dot(diff, diff);

            if (d2 < f1) {
                f2 = f1;
                f1 = d2;
                id = ts_hash2(wrapped, seed);  // cell identity, already unit range
            } else if (d2 < f2) {
                f2 = d2;
            }
        }
    }

    // Normalize: max possible squared distance to nearest feature in 3×3 search
    // with jitter=1 is bounded by 2.0.  sqrt then divide by sqrt(2) → [0,1].
    const float INV_SQRT2 = 0.70710678;
    TSWorley o;
    o.f1      = clamp(sqrt(f1) * INV_SQRT2, 0.0, 1.0);
    o.f2      = clamp(sqrt(f2) * INV_SQRT2, 0.0, 1.0);
    o.cell_id = id;
    return o;
}

float ts_white_tile(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    return ts_hash2(c, seed) * 2.0 - 1.0; // [-1, 1]
}

float ts_white_pcg(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    uvec3 h = ts_pcg3d(uvec3(uvec2(c), seed));
    return float(h.x) * (1.0 / 4294967295.0);
}

float ts_gabor_tile(vec2 p, int per, float freq, float bandwidth,
                    float anisotropy, float angle, uint seed) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    // Remap UI bandwidth [1..16] to Gaussian sigma. Higher UI bandwidth →
    // tighter kernel. We tie sigma to 'freq' so high-frequency kernels
    // get correspondingly tighter envelopes (Gabor uncertainty principle).
    float sigma   = 1.0 / max(bandwidth, 0.5);
    float two_s2  = 2.0 * sigma * sigma;
    float r2_max  = (2.5 * sigma) * (2.5 * sigma);  // hard cutoff at 2.5σ

    float sum = 0.0;
    anisotropy = clamp(anisotropy, 0.0, 1.0);

    // 5×5 search guarantees full envelope coverage and seamless tiling.
    for (int j = -2; j <= 2; ++j) {
        for (int i = -2; i <= 2; ++i) {
            ivec2 cell = ts_wrap2(pi + ivec2(i, j), per);

            // One pcg3d call → 3 decorrelated channels (kernel x, kernel y, angle)
            // instead of three separate ts_hash2 calls with seed offsets.
            vec3 h = ts_hash2_vec3(cell, seed);

            // Kernel position INSIDE the cell, [0,1)
            vec2 kpos = h.xy;
            // Vector from kernel center to sample point
            vec2 diff = vec2(i, j) + kpos - pf;
            float r2  = dot(diff, diff);
            if (r2 > r2_max) continue;

            // Anisotropic Gabor orientation
            float rand_angle = h.z * 6.28318530718;   // [0, 2π)
            float k_angle    = mix(rand_angle, angle, anisotropy);
            vec2  k_dir      = vec2(cos(k_angle), sin(k_angle));

            // Gaussian envelope × cosine carrier
            float env  = exp(-r2 / two_s2);
            float wave = cos(6.2831853 * freq * dot(diff, k_dir));
            sum += env * wave;
        }
    }

    // Empirical normalization: ~25 kernels × peak env ≈ 1.5 in worst case.
    return clamp(sum * 0.4, -1.0, 1.0);
}


// =============================================================================
// §11  FBM WRAPPERS
// =============================================================================
const float TS_OCTAVE_SEED_PRIME = 6791.0;
const uint  TS_OCTAVE_SEED_PRIME_U = 6791u;
const float TS_PI_OVER_PHI = 1.94161103873; // π / φ

// ---- Value FBM (period scales with lacunarity) ----
float ts_fbm_value(vec2 p, int period, float octaves,
                   float lacunarity, float gain, uint seed,
                   out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    vec2 gr;
    float base = ts_value_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_value_tile(p * freq, iper, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_value(vec2 p, int period, float octaves,
                   float lacunarity, float gain, uint seed) {
    vec2 g; return ts_fbm_value(p, period, octaves, lacunarity, gain, seed, g);
}

// ---- Perlin FBM ----
float ts_fbm_perlin(vec2 p, int period, float octaves,
                    float lacunarity, float gain, uint seed,
                    out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    vec2 gr;
    float base = ts_perlin_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_perlin_tile(p * freq, iper, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_perlin(vec2 p, int period, float octaves,
                    float lacunarity, float gain, uint seed) {
    vec2 g; return ts_fbm_perlin(p, period, octaves, lacunarity, gain, seed, g);
}

// ---- Simplex FBM (period scales uniformly — Y-period even-requirement
// satisfied by caller via ts_period_even) ----
float ts_fbm_simplex(vec2 p, ivec2 period, float octaves,
                     float lacunarity, float gain,
                     float alpha, uint seed,
                     out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int lac_int = int(lacunarity);
    ivec2 per_i = ivec2(
        max(period.x, 1),
        max(period.y, 1)
    );
    uint os = seed;
    float oct_alpha = alpha;
    vec2 gr;
    float base = ts_simplex_tile_seeded(p * freq, per_i, oct_alpha, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        per_i *= lac_int;
        oct_alpha = alpha + oct * TS_PI_OVER_PHI;
        os += TS_OCTAVE_SEED_PRIME_U;
        vec2 g2;
        float n = ts_simplex_tile_seeded(p * freq, per_i, oct_alpha, os, g2);
        base += weight * n;
        total_grad += weight * freq * g2;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    total_grad *= inv;
    return base * inv;
}

float ts_fbm_simplex(vec2 p, ivec2 period, float octaves,
                     float lacunarity, float gain,
                     float alpha, uint seed) {
    vec2 g;
    return ts_fbm_simplex(p, period, octaves, lacunarity, gain, alpha, seed, g);
}


TSWorley ts_fbm_worley(vec2 p, int period, float octaves,
                       float lacunarity, float gain, float jitter, uint seed) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    TSWorley base = ts_worley_tile(p * freq, iper, jitter, os);
    float f1_sum = base.f1, f2_sum = base.f2, id_sum = base.cell_id;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) {
        TSWorley o;
        o.f1 = clamp(f1_sum, 0.0, 1.0);
        o.f2 = clamp(f2_sum, 0.0, 1.0);
        o.cell_id = clamp(id_sum, 0.0, 1.0);
        return o;
    }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        TSWorley w2 = ts_worley_tile(p * freq, iper, jitter, os);
        f1_sum += weight * w2.f1;
        f2_sum += weight * w2.f2;
        id_sum += weight * w2.cell_id;
        norm += weight;
    } while (oct < octaves);
    float inv = 1.0 / max(norm, 1e-6);
    TSWorley o;
    o.f1 = clamp(f1_sum * inv, 0.0, 1.0);
    o.f2 = clamp(f2_sum * inv, 0.0, 1.0);
    o.cell_id = clamp(id_sum * inv, 0.0, 1.0);
    return o;
}

// ---- Gabor FBM ----
float ts_fbm_gabor(vec2 p, int period, float octaves,
                   float lacunarity, float gain,
                   float freq_hz, float bandwidth,
                   float anisotropy, float angle, uint seed) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    int iper = max(period, 1);
    uint os = seed;
    float base = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth, anisotropy, angle, os);
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return clamp(base, -1.0, 1.0); }
    int lac_int = int(lacunarity);
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        iper *= lac_int;
        os += TS_OCTAVE_SEED_PRIME_U;
        float n = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth, anisotropy, angle, os);
        base += weight * n;
        norm += weight;
    } while (oct < octaves);
    return clamp(base / max(norm, 1e-6), -1.0, 1.0);
}

// =============================================================================
// REMAP HELPERS
// =============================================================================
// All FBMs return [-1, 1].  want [0, 1].
float ts_to_unit(float n)  { return n * 0.5 + 0.5; }
vec2  ts_to_unit(vec2  n)  { return n * 0.5 + 0.5; }

#endif // TS_NOISE_COMMON

vec4 node_gabor(vec2 uv,
                float period,
                float octaves,
                float lacunarity,
                float roughness,
                float frequency,
                float bandwidth,
                float anisotropy,
                float angle,
                float speed,
                float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);
    float aniso = clamp(anisotropy, 0.0, 1.0);

    vec2 p = fract(uv) * float(iper)
           + vec2(pc.time * speed);

    float r = ts_fbm_gabor(p, iper, octaves, lacunarity, roughness,
                           frequency, bandwidth, aniso, angle, iseed);
    float g = ts_fbm_gabor(p, iper, octaves, lacunarity, roughness,
                           frequency, bandwidth, aniso, angle, iseed + 379u);
    float b = ts_fbm_gabor(p, iper, octaves, lacunarity, roughness,
                           frequency, bandwidth, aniso, angle, iseed + 757u);
    float a = ts_fbm_gabor(p, iper, octaves, lacunarity, roughness,
                           frequency, bandwidth, aniso, angle, iseed + 1013u);

    return vec4(ts_to_unit(r), ts_to_unit(g), ts_to_unit(b), ts_to_unit(a));
}



float ts_levels_single(float v, float in_low, float in_mid, float in_high,
                       float out_low, float out_high) {
    float range = in_high - in_low;
    float x;
    if (range >= 0.0) {
        x = clamp((v - in_low) / max(range, 1e-6), 0.0, 1.0);
    } else {
        x = clamp((in_low - v) / max(-range, 1e-6), 0.0, 1.0);
    }

    float t = clamp(in_mid, 1e-4, 1.0 - 1e-4);
    float p = log(0.5) / log(t);

    return pow(x, p) * (out_high - out_low) + out_low;
}

vec4 node_levels(vec2 uv, vec4 color,
                 float in_low_l, float in_mid_l, float in_high_l,
                 float out_low_l, float out_high_l,
                 float in_low_r, float in_mid_r, float in_high_r,
                 float out_low_r, float out_high_r,
                 float in_low_g, float in_mid_g, float in_high_g,
                 float out_low_g, float out_high_g,
                 float in_low_b, float in_mid_b, float in_high_b,
                 float out_low_b, float out_high_b,
                 float in_low_a, float in_mid_a, float in_high_a,
                 float out_low_a, float out_high_a,
                 float channel_mode)
{
    int m = int(channel_mode + 0.5);

    float r, g, b, a;

    if (m == 0) {
        // Luminance: L settings override R, G, B
        r = ts_levels_single(color.r, in_low_l, in_mid_l, in_high_l, out_low_l, out_high_l);
        g = ts_levels_single(color.g, in_low_l, in_mid_l, in_high_l, out_low_l, out_high_l);
        b = ts_levels_single(color.b, in_low_l, in_mid_l, in_high_l, out_low_l, out_high_l);
    } else {
        // Per-channel: each uses its own settings
        r = ts_levels_single(color.r, in_low_r, in_mid_r, in_high_r, out_low_r, out_high_r);
        g = ts_levels_single(color.g, in_low_g, in_mid_g, in_high_g, out_low_g, out_high_g);
        b = ts_levels_single(color.b, in_low_b, in_mid_b, in_high_b, out_low_b, out_high_b);
    }

    a = ts_levels_single(color.a, in_low_a, in_mid_a, in_high_a, out_low_a, out_high_a);

    return vec4(r, g, b, a);
}



#ifndef TS_COLOR_COMMON
#define TS_COLOR_COMMON

vec3 ts_rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)),
                d / (q.x + e), q.x);
}

vec3 ts_hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

#endif
#ifndef TS_BLEND_COMMON
#define TS_BLEND_COMMON
// requires ts_rgb2hsv / ts_hsv2rgb from color_common.glsl

// ---- per-channel scalar primitives ------------------------
float ts_b_add       (float a, float b) { return min(a + b, 1.0); }
float ts_b_avg       (float a, float b) { return (a + b) * 0.5; }
float ts_b_darken    (float a, float b) { return min(a, b); }
float ts_b_lighten   (float a, float b) { return max(a, b); }
float ts_b_diff      (float a, float b) { return abs(a - b); }
float ts_b_mul       (float a, float b) { return a * b; }
float ts_b_screen    (float a, float b) { return 1.0 - (1.0 - a) * (1.0 - b); }
float ts_b_excl      (float a, float b) { return a + b - 2.0 * a * b; }
float ts_b_negation  (float a, float b) { return 1.0 - abs(1.0 - a - b); }
float ts_b_subtract  (float a, float b) { return max(a + b - 1.0, 0.0); }
float ts_b_linBurn   (float a, float b) { return max(a + b - 1.0, 0.0); }
float ts_b_linDodge  (float a, float b) { return min(a + b, 1.0); }
float ts_b_colorBurn (float a, float b) { return (b == 0.0) ? b : max(1.0 - (1.0 - a) / b, 0.0); }
float ts_b_colorDodge(float a, float b) { return (b == 1.0) ? b : min(a / (1.0 - b), 1.0); }
float ts_b_overlay   (float a, float b) { return (a < 0.5) ? 2.0*a*b : 1.0 - 2.0*(1.0-a)*(1.0-b); }
float ts_b_hardLight (float a, float b) { return ts_b_overlay(b, a); }
float ts_b_softLight (float a, float b) {
    return (b < 0.5) ? (2.0*a*b + a*a*(1.0 - 2.0*b))
                     : (sqrt(a) * (2.0*b - 1.0) + 2.0*a*(1.0 - b));
}
float ts_b_reflect   (float a, float b) { return (b == 1.0) ? b : min(a*a/(1.0-b), 1.0); }
float ts_b_glow      (float a, float b) { return ts_b_reflect(b, a); }
float ts_b_harmony   (float a, float b) { return min(a,b) - max(a,b) + 1.0; }
float ts_b_linLight  (float a, float b) { return (b < 0.5) ? ts_b_linBurn (a, 2.0*b) : ts_b_linDodge (a, 2.0*(b-0.5)); }
float ts_b_vividLight(float a, float b) { return (b < 0.5) ? ts_b_colorBurn(a, 2.0*b) : ts_b_colorDodge(a, 2.0*(b-0.5)); }
float ts_b_pinLight  (float a, float b) { return (b < 0.5) ? ts_b_darken  (a, 2.0*b) : ts_b_lighten  (a, 2.0*(b-0.5)); }
float ts_b_hardMix   (float a, float b) { return (ts_b_vividLight(a,b) < 0.5) ? 0.0 : 1.0; }

// ---- vec3 lift macro --------------------------------------------------------
#define TS_BLEND_VEC3(FN) vec3(FN(a.r,b.r), FN(a.g,b.g), FN(a.b,b.b))

// ---- HSL-style modes (whole-color, not per-channel) -------------------------
vec3 ts_b_hue       (vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(B.x, A.y, A.z)); }
vec3 ts_b_saturation(vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(A.x, B.y, A.z)); }
vec3 ts_b_color     (vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(B.x, B.y, A.z)); }
vec3 ts_b_luminosity(vec3 a, vec3 b) { vec3 A=ts_rgb2hsv(a), B=ts_rgb2hsv(b); return ts_hsv2rgb(vec3(A.x, A.y, B.z)); }

#endif
// modes must match Enum order in blend.py

vec4 node_blend(vec2 uv, float mask, vec4 a, vec4 b, float mode) {
    int  m = int(mode + 0.5);
    float f = clamp(mask, 0.0, 1.0);
    vec3 r;
    if      (m ==  1) r = TS_BLEND_VEC3(ts_b_add);
    else if (m ==  2) r = TS_BLEND_VEC3(ts_b_mul);
    else if (m ==  3) r = TS_BLEND_VEC3(ts_b_screen);
    else if (m ==  4) r = TS_BLEND_VEC3(ts_b_overlay);
    else if (m ==  5) r = TS_BLEND_VEC3(ts_b_diff);
    else if (m ==  6) r = TS_BLEND_VEC3(ts_b_darken);
    else if (m ==  7) r = TS_BLEND_VEC3(ts_b_lighten);
    else if (m ==  8) r = TS_BLEND_VEC3(ts_b_colorBurn);
    else if (m ==  9) r = TS_BLEND_VEC3(ts_b_colorDodge);
    else if (m == 10) r = TS_BLEND_VEC3(ts_b_linBurn);
    else if (m == 11) r = TS_BLEND_VEC3(ts_b_linDodge);
    else if (m == 12) r = TS_BLEND_VEC3(ts_b_linLight);
    else if (m == 13) r = TS_BLEND_VEC3(ts_b_vividLight);
    else if (m == 14) r = TS_BLEND_VEC3(ts_b_pinLight);
    else if (m == 15) r = TS_BLEND_VEC3(ts_b_hardLight);
    else if (m == 16) r = TS_BLEND_VEC3(ts_b_softLight);
    else if (m == 17) r = TS_BLEND_VEC3(ts_b_hardMix);
    else if (m == 18) r = TS_BLEND_VEC3(ts_b_excl);
    else if (m == 19) r = TS_BLEND_VEC3(ts_b_subtract);
    else if (m == 20) r = TS_BLEND_VEC3(ts_b_avg);
    else if (m == 21) r = TS_BLEND_VEC3(ts_b_negation);
    else if (m == 22) r = TS_BLEND_VEC3(ts_b_reflect);
    else if (m == 23) r = TS_BLEND_VEC3(ts_b_glow);
    else if (m == 24) r = TS_BLEND_VEC3(ts_b_harmony);
    else if (m == 25) r = ts_b_hue       (a.rgb, b.rgb);
    else if (m == 26) r = ts_b_saturation(a.rgb, b.rgb);
    else if (m == 27) r = ts_b_color     (a.rgb, b.rgb);
    else if (m == 28) r = ts_b_luminosity(a.rgb, b.rgb);
    else              r = mix(a.rgb, b.rgb, f);

    if (m != 0) r = mix(a.rgb, r, f);
    return vec4(clamp(r, 0.0, 1.0), mix(a.a, b.a, f));
}


vec4 node_shuffle(vec2 uv, vec4 color,
                  float r_src, float g_src,
                  float b_src, float a_src)
{
    float sources[4] = float[4](color.r, color.g, color.b, color.a);
    return vec4(
        sources[int(r_src + 0.5)],
        sources[int(g_src + 0.5)],
        sources[int(b_src + 0.5)],
        sources[int(a_src + 0.5)]
    );
}



void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv;
    uv.x = float(coord.x) / float(max(pc.resolution_x, 1u));
    uv.y = float(coord.y) / float(max(pc.resolution_y, 1u));
    vec4 _result = vec4(0.0);

    if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;
    vec4 _local_0;
    _local_0 = node_worley(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6]);
    vec4 _local_1;
    _local_1 = node_worley(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6]);
    vec4 _local_2;
    _local_2 = node_simplex(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6]);
    vec4 _local_3;
    _local_3 = node_gabor(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 7], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 8], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 9]);
    vec4 _local_4;
    _local_4 = node_levels(uv, _local_0, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 7], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 8], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 9], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 10], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 11], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 12], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 13], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 14], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 15], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 16], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 17], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 18], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 19], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 20], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 21], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 22], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 23], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 24], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 25]);
    vec4 _local_5;
    _local_5 = node_levels(uv, _local_1, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 7], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 8], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 9], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 10], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 11], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 12], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 13], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 14], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 15], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 16], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 17], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 18], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 19], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 20], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 21], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 22], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 23], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 24], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 25]);
    vec4 _local_6;
    _local_6 = node_levels(uv, _local_2, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 7], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 8], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 9], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 10], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 11], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 12], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 13], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 14], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 15], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 16], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 17], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 18], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 19], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 20], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 21], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 22], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 23], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 24], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 25]);
    vec4 _local_7;
    _local_7 = node_levels(uv, _local_3, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 7], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 8], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 9], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 10], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 11], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 12], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 13], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 14], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 15], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 16], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 17], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 18], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 19], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 20], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 21], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 22], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 23], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 24], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 25]);
    vec4 _local_8;
    _local_8 = node_blend(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 1], _local_4, _local_5, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0]);
    vec4 _local_9;
    _local_9 = node_levels(uv, _local_8, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 6], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 7], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 8], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 9], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 10], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 11], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 12], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 13], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 14], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 15], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 16], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 17], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 18], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 19], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 20], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 21], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 22], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 23], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 24], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 25]);
    vec4 _local_10;
    _local_10 = node_blend(uv, _local_6.r, _local_9, _local_7, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0]);
    vec4 _local_11;
    _local_11 = node_shuffle(uv, _local_10, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3]);

    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, _local_11);
}
