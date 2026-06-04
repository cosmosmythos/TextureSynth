# Node Manifests

Each `*.node.json` describes a node type: its IO, parameters, shader source,
and Stage 2 classification. The 7 `pass_kind` values are listed in the table
below; see `DEV_LOG/IMPLEMENTATION_PLAN_pass_fusion/03_pass_kind.md` for the
full design and `DEV_LOG/IMPLEMENTATION_PLAN_pass_fusion/00a_DESIGN_DECISIONS.md`
for the chain-fusion rationale.

## `pass_kind` reference

| `pass_kind` value | Meaning | Chain-fuseable? | Stage 2 default |
|---|---|---|---|
| `pure_pixel` | 1 texel in, 1 texel out, no barriers/halo | **Yes** | yes (safe default) |
| `boundary` | Needs barriers, halo, resolution change, **or multi-output** | No (chain break) | |
| `reduction` | N texels in, 1 out (e.g. blur, mip downsample) | No (chain break) | |
| `feedback` | Reads its own previous output (temporal) | No (chain break) | |
| `upload` | CPU/external source -- end of a chain upstream, no dispatch | No (chain break) | |
| `readback` | GPU -> CPU -- terminal node, end of every chain | No (chain break) | |
| `debug_preview` | Materializes an intermediate for inspection | No (chain break) | |

A missing or unknown `pass_kind` is treated as `pure_pixel` (the safe
fuseable option) with a logged warning. Combiners and generators are
all `pure_pixel`; only `split_rgba` is `boundary` today (multi-output
is a chain break because the chain shader emits one `imageStore`).

## File map (13 manifests, current)

| Manifest | `pass_kind` | Notes |
|---|---|---|
| `perlin.node.json`     | `pure_pixel` | noise generator |
| `simplex.node.json`    | `pure_pixel` | noise generator |
| `value.node.json`      | `pure_pixel` | noise generator |
| `gabor.node.json`      | `pure_pixel` | noise generator |
| `worley.node.json`     | `pure_pixel` | noise generator |
| `white_noise.node.json`| `pure_pixel` | noise generator |
| `blend.node.json`      | `pure_pixel` | 2-vec4 combiner |
| `grayscale.node.json`  | `pure_pixel` | luminance collapse |
| `invert.node.json`     | `pure_pixel` | 1.0 - x |
| `combine_rgba.node.json` | `pure_pixel` | 4-in packer |
| `color_const.node.json`  | `pure_pixel` | constant source |
| `image.node.json`        | `pure_pixel` | sampler-bound; per-texel, no halo |
| `split_rgba.node.json`   | `boundary`   | **multi-output** -> chain break |

## GLSL function signature contract (mandatory for chain fusion)

**Every `*.glsl` file in this directory MUST follow this signature contract.** It is what makes chain fusion work.

### The 3 rules

1. **A `PurePixel` node's function takes `vec4 color` (or nothing for source nodes) and returns a `vec4`.** No `sampler2D`, no `imageLoad()`, no `texture()` inside a node function.
2. **A `Boundary` / `Reduction` / `Feedback` node MAY take a `sampler2D` for halo reads** (e.g., a future blur), but it is its own dispatch — never fused into a chain.
3. **The chain `main()` is the only place that touches VRAM** (one `imageLoad` at chain start, one `imageStore` at chain end).

### Why this matters

If a node function used `sampler2D` for its predecessor, the chain shader would have to re-read VRAM for every node step — defeating the whole optimization. The `vec4` style keeps the predecessor's output in a **thread-local register** (a `vec4` local variable), which is what makes `1 dispatch per chain` possible.

### Signature templates

```glsl
// Source node: no input, produces vec4
vec4 node_<name>(vec2 uv, /* params */);

// PurePixel node: 1 in, 1 out
vec4 node_<name>(vec2 uv, vec4 color, /* params */);

// PurePixel node: 2-in combiner (e.g. blend)
vec4 node_<name>(vec2 uv, vec4 a, vec4 b, /* params */);

// PurePixel node: 4-in packer (e.g. combine_rgba)
vec4 node_<name>(vec2 uv, vec4 r, vec4 g, vec4 b, vec4 a);

// Boundary node: 1 in, 4 out (e.g. split_rgba)
void node_<name>(vec2 uv, vec4 color,
                 out vec4 r_out, out vec4 g_out,
                 out vec4 b_out, out vec4 a_out);

// Reduction node (future): halo reads via sampler2D
vec4 node_<name>(vec2 uv, sampler2D input_image, int radius, /* params */);
```

### Parameter ordering

Always: `(vec2 uv, vec4 color_in, /* vec4 color_in2, ... */, <sliders>, <uniforms>)`.

- `vec2 uv` is always first (chain-uniform; same UV is fed to every node)
- `vec4` inputs follow (in topological order, no skipping)
- Sliders come last (these go through push constants, not the chain hash)
- Uniforms (textures, resolution) come after sliders

### Enforcement

- The 13 existing manifests already follow this contract (see `perlin.glsl`, `invert.glsl`, `grayscale.glsl`, `split_rgba.glsl`).
- Stage 3's chain emitter (`emit_chain_shader`, planned for Stage 4) will read each node's signature via the manifest's `glsl_signature` field (to be added) and refuse to emit a chain if a node uses `sampler2D` mid-chain.
