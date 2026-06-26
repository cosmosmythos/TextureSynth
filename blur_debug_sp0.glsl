#version 460
#define TS_FORMAT 3
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform texture2D u_sampled[];
layout(set = 0, binding = 1, rgba16f) writeonly uniform image2D u_storage[];
layout(set = 0, binding = 2) uniform sampler samp_repeat;
layout(set = 0, binding = 3) uniform sampler samp_clamp;
layout(set = 0, binding = 4) uniform sampler samp_mirror;
layout(set = 0, binding = 5, std430) readonly buffer NodeParams { float v[]; } node_params[3];

layout(push_constant, std430) uniform PC {
    uint  resolution_x;
    uint  resolution_y;
    uint  seed;
    float time;
    uint  out_storage_slots[4];
    uint  param_base_slot;
    uint  input_count;
    uint  param_ring_idx;
    uint  in_sampled_slots[8];
    uint  pass_index;
} pc;

layout(constant_id = 0) const uint ts_pass_index = 0u;

struct TSTexture { int slot; vec2 inv_size; };

vec4 ts_sample(int local, vec2 uv) {
    uint gslot = pc.in_sampled_slots[local];
    return texture(sampler2D(u_sampled[nonuniformEXT(gslot)], samp_repeat), uv);
}
vec4 ts_sample_lod(int local, vec2 uv, float lod) {
    uint gslot = pc.in_sampled_slots[local];
    return textureLod(sampler2D(u_sampled[nonuniformEXT(gslot)], samp_repeat), uv, lod);
}
ivec2 ts_size(int local) {
    uint gslot = pc.in_sampled_slots[local];
    return textureSize(u_sampled[nonuniformEXT(gslot)], 0);
}
vec4  Sample      (TSTexture t, vec2 uv)            { return ts_sample(t.slot, uv); }
vec4  SampleLevel (TSTexture t, vec2 uv, float lod) { return ts_sample_lod(t.slot, uv, lod); }
ivec2 GetSize     (TSTexture t)                     { return ts_size(t.slot); }
vec2  GetTexelSize(TSTexture t)                     { return t.inv_size; }

TSTexture ts_input_tex() {
    int li = 0;
    uint gslot = pc.in_sampled_slots[li];
    return TSTexture(li, 1.0 / vec2(textureSize(u_sampled[nonuniformEXT(gslot)], 0)));
}

vec4 node_blur(vec2 uv, TSTexture tex, float intensity) {
    if (intensity < 1e-5) return Sample(tex, uv);

    vec2 inv_res = 1.0 / vec2(GetSize(tex));
    float sigma  = intensity * 50.0;
    vec2 dir     = (ts_pass_index == 0u) ? vec2(sigma, 0.0) : vec2(0.0, sigma);

    vec2 off1 = 1.3846153846 * dir * inv_res;
    vec2 off2 = 3.2307692308 * dir * inv_res;

#define TSBLUR(o) texture(sampler2D(u_sampled[nonuniformEXT(pc.in_sampled_slots[0])], samp_repeat), uv + (o))

    return TSBLUR(vec2(0))  * 0.2270270270
         + TSBLUR(+off1)    * 0.3162162162
         + TSBLUR(-off1)    * 0.3162162162
         + TSBLUR(+off2)    * 0.0702702703
         + TSBLUR(-off2)    * 0.0702702703;

#undef TSBLUR
}



void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;
    vec2 uv;
    uv.x = float(coord.x) / float(max(pc.resolution_x, 1u));
    uv.y = float(coord.y) / float(max(pc.resolution_y, 1u));
    vec4 result = node_blur(uv, ts_input_tex(), node_params[pc.param_ring_idx].v[pc.param_base_slot + 0]);
    // Format post-process
#if TS_FORMAT == 0
    result = vec4(result.r, 0.0, 0.0, 1.0);
#elif TS_FORMAT == 1
    result = vec4(result.r, result.g, 0.0, 1.0);
#elif TS_FORMAT == 2
    result = vec4(result.r, result.g, result.b, 1.0);
#elif TS_FORMAT == 3
    result = vec4(result.r, result.g, result.b, result.a);
#else
    result = vec4(result.r, result.g, result.b, result.a);
#endif
    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, result);
}
