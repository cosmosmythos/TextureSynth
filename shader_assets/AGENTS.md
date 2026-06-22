# shader_assets — Node Manifests & GLSL

## Purpose
Declares the node library consumed by the engine at runtime. Two kinds of artifact:
1. `nodes/*.node.json` — per-node manifest: IO sockets, params, shader source, feature flags.
2. `nodes/*.glsl` — per-node function source (one function per file).
Plus shared GLSL includes under `glsl/`.

Loaded by `NodeRegistryLoader` (C++) at `Engine::init(nodes_dir, glsl_dir)`. The Blender addon's node factory reads the same manifests to generate Python node classes.

## Ownership
- `nodes/` — manifests + node GLSL + `README.md` (signature contract) + `format_nodes.{bat,py}` (clang-format runner).
- `glsl/` — shared includes: `noise_common.glsl`, `blend_common.glsl`, `color_common.glsl`, `sampling_common.glsl`, `remap.glsl`.
- `gather_glsl.bat`, `gather_json.bat` — collect sources for embedding/shipping.
- `Houdini_Generated.c` — generated Houdini export (large, vendored).
- `glslViewer_tests/` — standalone GLSL test inputs (not part of the engine pipeline).

## Local Contracts
- **Signature contract** (`nodes/README.md`): every node function is
  `vec4 node_<name>(vec2 uv, /* vec4 inputs... */, /* sliders */, /* uniforms */)`.
  - Source nodes: take only `vec2 uv` (+ params).
  - Combiners: take multiple `vec4`.
  - Multi-output (e.g. `separate_rgba`): use `out vec4` parameters.
  - **No** `sampler2D`, `imageLoad()`, or `texture()` inside a node function. Image sampling is handled by the engine's image-upload path, not the node fn.
- **Parameter ordering** (mandatory): `(vec2 uv, vec4 color_in[, color_in2...], <sliders>, <uniforms>)`. Sliders feed push constants; uniforms (textures, resolution) come last.
- **`pass_kind`** (`nodes/README.md` §pass_kind): `compute` (default), `upload` (CPU→GPU source), `readback` (GPU→CPU terminal). Old values (`pure_pixel`, `boundary`, `reduction`, `feedback`, `debug_preview`) map to `compute` for back-compat.
- **`variant_flags`** in a manifest → `NodeType::variant_flags` → `ShaderVariantKey` specialization constants. Only set on nodes that genuinely need compile-time variants.
- **`format_sensitive`**: ONLY noise generators (perlin, simplex, value, gabor, worley, white_noise) set this. Combiners and constant sources MUST NOT, or the Mono path collapses their output. See `Graph.hpp:107-115`.
- **Manifest ↔ factory parity**: adding a node means (1) new `*.node.json` + `*.glsl` here, (2) the engine picks it up automatically via `NodeRegistryLoader`, (3) the Blender factory generates a class automatically. No C++ or Python node code for standard nodes.

## Work Guidance
- Format GLSL with `nodes/format_nodes.bat` (wraps clang-format; config in `format_nodes.py`).
- Noise primitives live in `glsl/noise_common.glsl` (PCG hash family, gradient tables, etc.). New noise nodes should reuse these, not re-derive hashes.
- When editing a common include, grep all `nodes/*.glsl` for `#include`/usage — many node fns depend on the shared headers.

## Verification
- Engine loads manifests at init; a missing/malformed manifest surfaces as `EngineError` with the offending node id.
- `tests/test_node_library.py` (Python) and `tests/test_noise_nodes.cpp` (C++) exercise specific nodes.
- Standalone GLSL sanity: files under `glslViewer_tests/` can be run in glslViewer (not automated by CI).

## Child DOX Index
None — `nodes/` and `glsl/` are cohesive; the `nodes/README.md` is the detailed contract and should stay the canonical reference (this doc points to it, does not duplicate it).
