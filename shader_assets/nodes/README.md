# Node Manifests

Each `*.node.json` describes a node type: its IO, parameters, and shader source.

## `pass_kind` reference

| `pass_kind` value | Meaning | Default? |
|---|---|---|
| `compute` | Has a compute shader — dispatched normally | yes (default when omitted) |
| `upload` | CPU → GPU source — no shader, no dispatch | |
| `readback` | GPU → CPU terminal — no shader, no dispatch | |

A missing `pass_kind` defaults to `compute`. Old values (`pure_pixel`,
`boundary`, `reduction`, `feedback`, `debug_preview`) are accepted
backwards-compatibly and mapped to `compute`.

## File map (16 manifests)

All 16 manifests use the default `compute` — no explicit `pass_kind` key
is needed.

| Manifest | Notes |
|---|---|
| `perlin.node.json`     | noise generator |
| `simplex.node.json`    | noise generator |
| `value.node.json`      | noise generator |
| `gabor.node.json`      | noise generator |
| `worley.node.json`     | noise generator |
| `white_noise.node.json`| noise generator |
| `blend.node.json`      | 2-vec4 combiner |
| `grayscale.node.json`  | luminance collapse |
| `invert.node.json`     | 1.0 - x |
| `combine_rgba.node.json` | 4-in packer |
| `color_const.node.json`  | constant source |
| `image.node.json`        | sampler-bound; per-texel |
| `separate_rgba.node.json` | multi-output |
| `remap.node.json`        | range remap |
| `blur.node.json`         | Gaussian blur; sampler-bound |
| `warp.node.json`         | gradient-based UV displacement; sampler-bound |

## GLSL function signature contract

**Every `*.glsl` file in this directory MUST follow this signature contract.**

### The rules

1. **A node's function takes `vec2 uv` first, then `vec4` inputs, then params, and returns `vec4`.** No `sampler2D`, no `imageLoad()`, no `texture()` inside a node function.
2. **Source nodes** take only `vec2 uv` (and params).
3. **Combiners** take multiple `vec4` inputs.

### Signature templates

```glsl
// Source node: no input, produces vec4
vec4 node_<name>(vec2 uv, /* params */);

// 1-input node (e.g. invert, grayscale)
vec4 node_<name>(vec2 uv, vec4 color, /* params */);

// 2-input combiner (e.g. blend)
vec4 node_<name>(vec2 uv, vec4 a, vec4 b, /* params */);

// 4-input packer (e.g. combine_rgba)
vec4 node_<name>(vec2 uv, vec4 r, vec4 g, vec4 b, vec4 a);

// Multi-output (e.g. separate_rgba)
void node_<name>(vec2 uv, vec4 color,
                 out vec4 r_out, out vec4 g_out,
                 out vec4 b_out, out vec4 a_out);
```

### Parameter ordering

Always: `(vec2 uv, vec4 color_in, /* vec4 color_in2, ... */, <sliders>, <uniforms>)`.

- `vec2 uv` is always first (same UV is fed to every node)
- `vec4` inputs follow (in topological order, no skipping)
- Sliders come last (these go through push constants)
- Uniforms (textures, resolution) come after sliders
