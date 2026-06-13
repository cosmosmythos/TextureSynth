#ifndef TS_SAMPLING_COMMON
#define TS_SAMPLING_COMMON

struct TSTexture {
    int  slot;        // 0..N-1; engine generates ts_dispatch_* dispatch
    vec2 inv_size;    // 1.0 / textureSize, for derivative math
};

vec4 ts_dispatch_sample(int slot, vec2 uv);
vec4 ts_dispatch_sample_lod(int slot, vec2 uv, float lod);
ivec2 ts_dispatch_size(int slot);

vec4  Sample      (TSTexture t, vec2 uv)            { return ts_dispatch_sample(t.slot, uv); }
vec4  SampleLevel (TSTexture t, vec2 uv, float lod) { return ts_dispatch_sample_lod(t.slot, uv, lod); }
ivec2 GetSize     (TSTexture t)                     { return ts_dispatch_size(t.slot); }
vec2  GetTexelSize(TSTexture t)                     { return t.inv_size; }

#endif
