#version 460

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform texture2D u_sampled[];
layout(set = 0, binding = 1, rgba16f) writeonly uniform image2D u_storage[];

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
vec4 _fmt_rgba(vec4 v) { return v; }
vec4 _fmt_id(vec4 v) { return v; }
vec4 _fmt_metadata(vec4 v) { return v; }


vec4 node_color_const(vec2 uv, float mode, float r, float g, float b, float a) {
    if (mode < 0.5) return vec4(r, r, r, 1.0);
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


void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv;
    uv.x = float(coord.x) / float(max(pc.resolution_x, 1u));
    uv.y = float(coord.y) / float(max(pc.resolution_y, 1u));
    vec4 _result = vec4(0.0);

    if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;
    vec4 _local_0;
    _local_0 = node_color_const(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 0 + 4]);
    vec4 _local_1;
    _local_1 = node_color_const(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 5 + 0], node_params[pc.param_ring_idx].v[pc.param_base_slot + 5 + 1], node_params[pc.param_ring_idx].v[pc.param_base_slot + 5 + 2], node_params[pc.param_ring_idx].v[pc.param_base_slot + 5 + 3], node_params[pc.param_ring_idx].v[pc.param_base_slot + 5 + 4]);
    vec4 _local_2;
    _local_2 = node_blend(uv, node_params[pc.param_ring_idx].v[pc.param_base_slot + 11], _local_0, _local_1, node_params[pc.param_ring_idx].v[pc.param_base_slot + 10 + 0]);

    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, _local_2);
}
