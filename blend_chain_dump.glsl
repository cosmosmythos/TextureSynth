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
layout(constant_id = 0) const uint ts_pass_index = 0u;
layout(push_constant, std430) uniform PC {
    uint resolution_x, resolution_y, seed;
    float time;
    uint out_storage_slots[4];
    uint param_base_slot;
    uint input_count;
    uint param_ring_idx;
    uint in_sampled_slots[8];
    uint pass_index;
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


vec4 node_color_const(vec2 uv, float mode, float r, float g, float b, float a) {
    if (mode < 0.5) return vec4(r, r, r, 1.0);
    return vec4(r, g, b, a);
}


#ifndef TS_BLEND_COMMON
#define TS_BLEND_COMMON

// a = foreground, b = background. Each fn returns the raw mode result.
#define TS_BLEND_VEC3(FN) vec3(FN(a.r,b.r), FN(a.g,b.g), FN(a.b,b.b))

float ts_blend_add (float a, float b) { return a + b; }
float ts_blend_sub (float a, float b) { return a - b; }
float ts_blend_mul (float a, float b) { return (a < 0.0 && b < 0.0) ? a : a * b; }
float ts_blend_min (float a, float b) { return min(a, b); }
float ts_blend_max (float a, float b) { return max(a, b); }
float ts_blend_avg (float a, float b) { return (a + b) * 0.5; }

float ts_blend_color_burn(float a, float b) { return 1.0 - (1.0 - b) / max(a, 1.0 - b); }

float ts_blend_overlay(float a, float b) {
    if (a < 0.0 || b < 0.0) return a;
    return (2.0 * b < 1.0) ? 2.0 * a * b : 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
}

float ts_blend_screen(float a, float b) {
    return (a > 0.0 && a < 1.0 && b > 0.0 && b < 1.0) ? a + b - a * b : max(a, b);
}

float ts_blend_color_dodge(float a, float b) { return b / max(1.0 - a, b); }

float ts_blend_soft_light(float a, float b) {
    return (a * b < 1.0) ? 2.0 * a * b + b * (1.0 - a * b) : 2.0 * a * b;
}

float ts_blend_hard_light(float a, float b) {
    if (2.0 * a < 1.0) return 2.0 * a * b;
    if (a < 1.0 && b < 1.0) return 1.0 - 2.0 * (1.0 - a) * (1.0 - b);
    return 0.0;
}

float ts_blend_divide(float a, float b) { return (b > 0.0) ? a / b : a; }
float ts_blend_diff (float a, float b) { return abs(a - b); }

float ts_blend_exclusion(float a, float b) {
    if (a < 0.0 || b < 0.0) return a;
    return a + b - 2.0 * a * b;
}

#endif

// a = foreground, b = background. mask: 0 -> B, 1 -> mode result.

vec4 node_blend(vec2 uv, float mask, vec4 a, vec4 b, float mode) {
    int m = int(mode + 0.5);
    float f = clamp(mask, 0.0, 1.0);
    vec3 r = a.rgb;
    switch (m) {
        case 1:  r = TS_BLEND_VEC3(ts_blend_add);          break;
        case 2:  r = TS_BLEND_VEC3(ts_blend_sub);          break;
        case 3:  r = TS_BLEND_VEC3(ts_blend_mul);          break;
        case 4:  r = TS_BLEND_VEC3(ts_blend_min);          break;
        case 5:  r = TS_BLEND_VEC3(ts_blend_max);          break;
        case 6:  r = TS_BLEND_VEC3(ts_blend_avg);          break;
        case 7:  r = TS_BLEND_VEC3(ts_blend_color_burn);   break;
        case 8:  r = TS_BLEND_VEC3(ts_blend_overlay);      break;
        case 9:  r = TS_BLEND_VEC3(ts_blend_screen);       break;
        case 10: r = TS_BLEND_VEC3(ts_blend_color_dodge);  break;
        case 11: r = TS_BLEND_VEC3(ts_blend_soft_light);   break;
        case 12: r = TS_BLEND_VEC3(ts_blend_hard_light);   break;
        case 13: r = TS_BLEND_VEC3(ts_blend_divide);       break;
        case 14: r = TS_BLEND_VEC3(ts_blend_diff);         break;
        case 15: r = TS_BLEND_VEC3(ts_blend_exclusion);    break;
    }
    return vec4(mix(b.rgb, r, f), mix(b.a, a.a, f));
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
