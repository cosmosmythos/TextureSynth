#ifndef TS_SAMPLING_COMMON
#define TS_SAMPLING_COMMON

// =============================================================================
// TS SAMPLING COMMON  —  Typed Texture-Input Convention
// =============================================================================
// Every node input of type sampler2D becomes a TSTexture handle.
// Authors NEVER call texture()/texelFetch() directly — always Sample()/...
// This abstracts:
//   • Vulkan separate sampler/image vs combined image-sampler
//   • Future LOD bias, anisotropy, clamp-mode variants
//   • Engine-side immutable sampler binding (slots 10/11/12)
// =============================================================================

struct TSTexture {
    int  slot;        // 0..N-1; engine generates ts_dispatch_* dispatch
    vec2 inv_size;    // 1.0 / textureSize, for derivative math
};

// Engine emits one of these per declared input:
//   TSTexture ts_input_tex() { return TSTexture(0, vec2(1.0/W, 1.0/H)); }
// Authors call:
//   vec4 c = Sample(ts_input_tex(), uv);

// Forward declarations — the engine generates the bodies per-graph.
vec4 ts_dispatch_sample(int slot, vec2 uv);
vec4 ts_dispatch_sample_lod(int slot, vec2 uv, float lod);
ivec2 ts_dispatch_size(int slot);

// Author-facing API:
vec4  Sample      (TSTexture t, vec2 uv)            { return ts_dispatch_sample(t.slot, uv); }
vec4  SampleLevel (TSTexture t, vec2 uv, float lod) { return ts_dispatch_sample_lod(t.slot, uv, lod); }
ivec2 GetSize     (TSTexture t)                     { return ts_dispatch_size(t.slot); }
vec2  GetTexelSize(TSTexture t)                     { return t.inv_size; }

#endif
