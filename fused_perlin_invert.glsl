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




vec4 _fmt_mono(vec4 v) { return vec4(v.x, v.x, v.x, 1.0); }
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


float ts_perm(float x) {
    return mod((34.0 * x + 10.0) * x, 289.0);
}

vec3 ts_perm3(vec3 x) {
    return mod((34.0 * x + 10.0) * x, 289.0);
}

vec4 ts_perm4(vec4 x) {
    return mod((34.0 * x + 10.0) * x, 289.0);
}

// Strong seed mixing — runs the seed through one round of the polynomial
// before combining with coordinates.
float ts_seed_mix(uint seed) {
    // Two rounds of integer multiply-shift (Wang-style), then map to [0,289).
    uint s = seed;
    s = (s ^ 61u) ^ (s >> 16u);
    s = s + (s << 3u);
    s = s ^ (s >> 4u);
    s = s * 0x27d4eb2du;
    s = s ^ (s >> 15u);
    return float(s & 0xFFFFu) * (1.0 / 65536.0) * 289.0;
}

// 2-D hash on integer lattice coordinates.  Returns [0, 288].
float ts_hash2(ivec2 c, uint seed) {
    float s = ts_seed_mix(seed);
    float h = ts_perm(float(c.x) + s);
    return ts_perm(h + float(c.y));
}

// Vectorized 4-corner hash (faster than 4 scalar calls — one SIMD path).
vec4 ts_hash2_quad(ivec2 c00, uint seed) {
    float s = ts_seed_mix(seed);
    vec4 xs = vec4(c00.x, c00.x + 1, c00.x,     c00.x + 1) + s;
    vec4 ys = vec4(c00.y, c00.y,     c00.y + 1, c00.y + 1);
    return ts_perm4(ts_perm4(xs) + ys);
}


vec2 ts_grad8(float hash) {
    float a = floor(mod(hash, 8.0)) * 0.78539816339; // π/4
    return vec2(cos(a), sin(a));
}

vec2 ts_grad16(float hash) {
    float a = floor(mod(hash, 16.0)) * 0.39269908170; // π/8
    return vec2(cos(a), sin(a));
}

// Continuous-angle gradient (for simplex / Gabor).  Uses the full hash range
// for maximum directional decorrelation.
vec2 ts_grad_continuous(float hash) {
    float a = hash * 0.07482; // (2π × ~3.44) / 289, well-distributed
    return vec2(cos(a), sin(a));
}


// =============================================================================
// §3  SMOOTHSTEP 
// =============================================================================
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


// =============================================================================
// §4  PERIOD
// =============================================================================
int ts_period_int(float p) {
    return max(int(round(p)), 1);
}

// Snap period to nearest power of 2 for perfect tiling (Gabor).
// Cosine carrier completes integer cycles at power-of-2 periods.
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

// Integer modulo that handles negative coordinates (GLSL % truncates).
int ts_wrap(int v, int per) {
    return ((v % per) + per) % per;
}

ivec2 ts_wrap2(ivec2 v, int per) {
    return ivec2(ts_wrap(v.x, per), ts_wrap(v.y, per));
}


// =============================================================================
// §5  VALUE NOISE  (tileable, analytical derivatives)
// =============================================================================
float ts_value_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,                 per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0),   per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1),   per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1),   per);

    // Corner values: hash → [-1, 1]
    float v00 = ts_hash2(c00, seed) * (2.0 / 288.0) - 1.0;
    float v10 = ts_hash2(c10, seed) * (2.0 / 288.0) - 1.0;
    float v01 = ts_hash2(c01, seed) * (2.0 / 288.0) - 1.0;
    float v11 = ts_hash2(c11, seed) * (2.0 / 288.0) - 1.0;

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


// =============================================================================
// §6  CLASSIC PERLIN NOISE  (tileable, analytical derivatives)
// =============================================================================
const float TS_PERLIN_NORM = 1.3393713;   // = 1.0 / 0.74682413

float ts_perlin_tile(vec2 p, int per, uint seed, out vec2 grad) {
    ivec2 pi = ivec2(floor(p));
    vec2  pf = p - vec2(pi);

    ivec2 c00 = ts_wrap2(pi,                 per);
    ivec2 c10 = ts_wrap2(pi + ivec2(1, 0),   per);
    ivec2 c01 = ts_wrap2(pi + ivec2(0, 1),   per);
    ivec2 c11 = ts_wrap2(pi + ivec2(1, 1),   per);

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


// =============================================================================
// §7  SIMPLEX NOISE  (psrdnoise2 — Gustavson & McEwan, JCGT 2022)
// =============================================================================
const float TS_SIMPLEX_NORM = 10.9;

float ts_simplex_tile(vec2 x, ivec2 period, float alpha, out vec2 gradient) {
    // Skew to simplex coordinate space
    vec2 uv = vec2(x.x + x.y * 0.5, x.y);

    vec2 i0 = floor(uv);
    vec2 f0 = fract(uv);

    float cmp = step(f0.y, f0.x);
    vec2 o1 = vec2(cmp, 1.0 - cmp);

    vec2 i1 = i0 + o1;
    vec2 i2 = i0 + vec2(1.0);

    // Back-transform corners to texture space
    vec2 v0 = vec2(i0.x - i0.y * 0.5,        i0.y       );
    vec2 v1 = vec2(v0.x + o1.x - o1.y * 0.5, v0.y + o1.y);
    vec2 v2 = vec2(v0.x + 0.5,               v0.y + 1.0 );

    vec2 x0 = x - v0;
    vec2 x1 = x - v1;
    vec2 x2 = x - v2;

    // Wrap corners to period (mandatory — no fast path; artists never disable tiling)
    vec3 xw = vec3(v0.x, v1.x, v2.x);
    vec3 yw = vec3(v0.y, v1.y, v2.y);
    xw = mod(xw, float(period.x));
    yw = mod(yw, float(period.y));
    vec3 iu = floor(xw + 0.5 * yw + 0.5);
    vec3 iv = floor(yw + 0.5);

    // Seed-mixed double-permutation hash
    float s = ts_seed_mix(0u);  // seed handled by caller via alpha offset
    vec3 h = mod(iu + s, 289.0);
    h = mod((h * 51.0 + 2.0) * h + iv, 289.0);
    h = mod((h * 34.0 + 10.0) * h,     289.0);

    vec3 psi = h * 0.07482 + alpha;
    vec3 gx  = cos(psi);
    vec3 gy  = sin(psi);

    vec2 g0 = vec2(gx.x, gy.x);
    vec2 g1 = vec2(gx.y, gy.y);
    vec2 g2 = vec2(gx.z, gy.z);

    // Radial decay kernel
    vec3 w  = max(0.8 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
    vec3 w2 = w * w;
    vec3 w4 = w2 * w2;

    vec3 gdotx = vec3(dot(g0,x0), dot(g1,x1), dot(g2,x2));
    float n = dot(w4, gdotx);

    // Analytical gradient
    vec3 w3 = w2 * w;
    vec3 dw = -8.0 * w3 * gdotx;
    vec2 dn0 = w4.x * g0 + dw.x * x0;
    vec2 dn1 = w4.y * g1 + dw.y * x1;
    vec2 dn2 = w4.z * g2 + dw.z * x2;

    gradient = TS_SIMPLEX_NORM * (dn0 + dn1 + dn2);
    return    TS_SIMPLEX_NORM * n;
}

// Seed-aware wrapper: folds seed into alpha as a base rotation.
float ts_simplex_tile_seeded(vec2 x, ivec2 period, float alpha, uint seed,
                             out vec2 gradient) {
    float seed_alpha = ts_seed_mix(seed) * 0.02174; // map [0,289) → [0, 2π)
    return ts_simplex_tile(x, period, alpha + seed_alpha, gradient);
}


// =============================================================================
// §8  WORLEY / CELLULAR NOISE  (tileable, F1, F2, F2-F1, cell IDs)
// =============================================================================
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

            // Two independent hashes for x and y offset within the cell
            float hx = ts_hash2(wrapped, seed);
            float hy = ts_hash2(wrapped, seed + 0x9E3779B9u); // golden-ratio offset

            vec2 feature = vec2(hx, hy) * (1.0 / 288.0); // [0, 1]
            feature = 0.5 + (feature - 0.5) * jitter;    // shrink toward cell center

            vec2 diff = vec2(i, j) + feature - pf;
            float d2 = dot(diff, diff);

            if (d2 < f1) {
                f2 = f1;
                f1 = d2;
                id = hx * (1.0 / 288.0); // cell identity
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


// =============================================================================
// §1b  PCG HASH  —  Integer Hash for White Noise
// =============================================================================
uvec2 ts_pcg2d(uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    return v;
}
// =============================================================================
// §9  WHITE NOISE  (per-pixel hash, no interpolation)
// =============================================================================
float ts_white_tile(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    return ts_hash2(c, seed) * (1.0 / 288.0) * 2.0 - 1.0; // [-1, 1]
}



// PCG white noise — full 32-bit palette, zero banding.
// Output: [0, 1].  Replaces ts_white_tile in the White Noise node.
float ts_white_pcg(vec2 p, int per, uint seed) {
    ivec2 c = ts_wrap2(ivec2(floor(p)), per);
    // Fold seed into both axes so different seeds produce unrelated fields.
    uvec2 v = uvec2(uint(c.x) ^ (seed * 2654435761u),
                    uint(c.y) ^ (seed * 2246822519u));
    uvec2 h = ts_pcg2d(v);
    return float(h.x) * (1.0 / 4294967295.0);
}


// =============================================================================
// §10  GABOR NOISE  (anisotropic, frequency-controlled, tileable)
// =============================================================================
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

            float h1 = ts_hash2(cell, seed);
            float h2 = ts_hash2(cell, seed + 0x9E3779B9u);
            float h3 = ts_hash2(cell, seed + 0x12345678u);

            // Kernel position INSIDE the cell, [0,1)
            vec2 kpos = vec2(h1, h2) * (1.0 / 288.0);
            // Vector from kernel center to sample point
            vec2 diff = vec2(i, j) + kpos - pf;
            float r2  = dot(diff, diff);
            if (r2 > r2_max) continue;

            // Anisotropic Gabor orientation
            float rand_angle = h3 * 0.02174;          // [0, 2π)
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

// ---- Value FBM (Houdini-matched octave blending) ----
float ts_fbm_value(vec2 p, int period, float octaves,
                   float lacunarity, float gain, uint seed,
                   out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0, fper = float(period);
    int iper = max(int(round(fper)), 1);
    uint os = seed;
    vec2 gr;
    float base = ts_value_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        fper *= lacunarity;
        iper = max(int(round(fper)), 1);
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

// ---- Perlin FBM (Houdini-matched octave blending) ----
float ts_fbm_perlin(vec2 p, int period, float octaves,
                    float lacunarity, float gain, uint seed,
                    out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0, fper = float(period);
    int iper = max(int(round(fper)), 1);
    uint os = seed;
    vec2 gr;
    float base = ts_perlin_tile(p * freq, iper, os, gr);
    total_grad = gr;
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return base; }
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        fper *= lacunarity;
        iper = max(int(round(fper)), 1);
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

// ---- Simplex FBM (Houdini-matched octave blending) ----
float ts_fbm_simplex(vec2 p, ivec2 period, float octaves,
                     float lacunarity, float gain,
                     float alpha, uint seed,
                     out vec2 total_grad) {
    float weight = 1.0;
    float g = gain * min(lacunarity, 1.0);
    float freq = 1.0;
    vec2 fper = vec2(period);
    ivec2 per_i = ivec2(
        max(int(round(fper.x)), 1),
        max(int(round(fper.y * 0.5)) * 2, 2)
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
        fper *= lacunarity;
        per_i = ivec2(
            max(int(round(fper.x)), 1),
            max(int(round(fper.y * 0.5)) * 2, 2)
        );
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
    float freq = 1.0, fper = float(period);
    int iper = max(int(round(fper)), 1);
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
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        fper *= lacunarity;
        iper = max(int(round(fper)), 1);
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
    float freq = 1.0, fper = float(period);
    int iper = max(int(round(fper)), 1);
    uint os = seed;
    float base = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth,
                               anisotropy, angle, os);
    float norm = 1.0;
    float oct = 1.0;
    if (oct >= octaves) { return clamp(base, -1.0, 1.0); }
    do {
        weight *= g;
        oct += 1.0;
        if (oct >= octaves) { weight *= 1.0 - (oct - octaves); }
        freq *= lacunarity;
        fper *= lacunarity;
        iper = max(int(round(fper)), 1);
        os += TS_OCTAVE_SEED_PRIME_U;
        float n = ts_gabor_tile(p * freq, iper, freq_hz, bandwidth,
                                anisotropy, angle, os);
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
vec4 node_perlin(vec2 uv,
                 float period,
                 float octaves,
                 float lacunarity,
                 float roughness,
                 float speed,
                 float seed)
{
    uint iseed = uint(clamp(seed, 0.0, 65535.0)) ^ pc.seed;
    int iper = ts_period_int(period);

    vec2 p = uv * float(iper)
           + vec2(pc.time * speed);

    float r = ts_fbm_perlin(p, iper, octaves, lacunarity, roughness, iseed);
    float g = ts_fbm_perlin(p, iper, octaves, lacunarity, roughness, iseed + 379u);
    float b = ts_fbm_perlin(p, iper, octaves, lacunarity, roughness, iseed + 757u);

    return vec4(ts_to_unit(r), ts_to_unit(g), ts_to_unit(b), 1.0);
}



vec4 node_invert(vec2 uv, float mask, vec4 color) {
    vec3 inv = vec3(1.0) - color.rgb;
    vec4 result = vec4(inv, color.a);
    return mix(color, result, clamp(mask, 0.0, 1.0));
}



void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(coord) / vec2(pc.resolution_x, pc.resolution_y);
    vec4 _result = vec4(0.0);

    if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;
    vec4 _local_0;
    _local_0 = node_perlin(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 5]);
    vec4 _local_1;
    _local_1 = node_invert(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 6], _local_0);

    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, _local_1);
}
