# src/engine/graphfusion — Chain Fusion Subsystem

## Purpose
Fuses connected compute nodes into single-dispatch groups so a subgraph runs in one shader instead of N. Header-only module (no `.cpp` files). Grouping, GLSL emission, and compilation are all inline in `.hpp` files under `te::fusion` / `te::glsl`.

## Ownership
All of `src/engine/graphfusion/` (4 files, all `.hpp`):
- `GlslBuilder.hpp` — `te::glsl::GlslBuilder`: string builder for fused compute shader GLSL (headers, layouts, functions, `main()` body with `imageStore`/`texelFetch`/bindless sampler patterns). Also owns `compute_header()` and `format_helpers()`.
- `FusionGroup.hpp` — `te::fusion`: `FusionGroup`, `FusionGroupBundle`, `FusionContext`, `ExpandedNode` structs + pipeline functions: `build_context()`, `expand_multipass()`, `group_nodes()`, `merge_groups()`, `compute_external_inputs()`, `get_connection_type()`, `is_connected()`, `group_contains()`.
- `FusionGroupEmitter.hpp` — `te::fusion::emit_group()`: takes one `FusionGroup` + `FusionContext` → `GroupEmitResult` with compiled GLSL source string.
- `FusedGroupCompiler.hpp` — `te::fusion`: `CompiledGroup`, `CompiledGroupBundle` structs + `compile_groups()`: top-level entry point, iterates groups and calls `emit_group()` for each.

**Deleted files (no longer in this folder):** `DAG.hpp`, `FusionPlanner.hpp`, `RegisterAllocator.hpp`, `FusedGraphEmitter.{hpp,cpp}`, `FusedGraphCompiler.{hpp,cpp}`.

## Local Contracts
- **Namespace**: `te::glsl` (GlslBuilder), `te::fusion` (everything else).
- **Header-only**: all functions are `inline` or defined in-class. No translation units — include from consumer code directly.
- **Bit-identity contract**: fused output must equal the unfused per-pass output for each node. Tests in `tests/test_fused_*.cpp` enforce this; do not land a fusion change that breaks `test_fused_real_nodes.cpp`.
- **GLSL contract**: emitted code calls node functions following `shader_assets/nodes/README.md` (`vec4 node_<name>(vec2 uv, ...)`). No `sampler2D` / `imageLoad` inside emitted node calls. External textures use bindless `TSTexture` struct with `Sample()`/`SampleLevel()`/`GetSize()`/`GetTexelSize()` helpers.
- **Pipeline** (entry → exit):
  1. `build_context(ir, lib)` → `FusionContext` (node types, connection lookups, param layout)
  2. `group_nodes(ir, ctx)` → `FusionGroupBundle` (initial grouping by connectivity, multi-pass expansion)
  3. `merge_groups(bundle, ctx)` — merges adjacent Vec4-connected groups, skips Sampler2D boundaries
  4. `compute_external_inputs(bundle, ctx)` — populates `FusionGroup::external_inputs` (cross-group Sampler2D connections → bindless slot indices)
  5. `compile_groups(bundle, ir, ctx)` → `CompiledGroupBundle` (calls `emit_group()` per group, collects GLSL + param layout + external inputs)
- **Grouping logic**: `group_nodes()` expands multi-pass nodes (`expand_multipass()`), then splits at disconnected or Sampler2D-connected pairs. `merge_groups()` recombines adjacent groups when the connection is Vec4 (not Sampler2D). Multi-pass nodes (`pass_count > 1`) expand into separate passes and stay isolated.
- **External inputs**: only `Sampler2D` connections crossing group boundaries become `external_inputs`. Vec4 cross-group connections are not tracked here (handled by the engine's chain slot system).
- **Param layout**: `FusionContext::param_base` maps each node to its SSBO float offset. `compile_groups()` computes `CompiledGroup::param_base_slot` (min offset across group) and `param_floats` (total floats).
- **Deprecated**: `FusionGroupEmitter.hpp:43` calls `glsl::compute_header()` without a `StorageFormat` argument, using the F32-only shim. New callers should pass `StorageFormat` explicitly.

## Work Guidance
- When adding a node that should fuse: confirm `NodeType::pass_kind == Compute` and its GLSL is pure (no side effects, no texture fetches inside the node fn). Image nodes (`Sampler2D` input) ARE fusible.
- Debug fused vs. unfused divergence with `tests/compare_fusion_outputs.bat` and `test_fused_real_nodes.cpp`.
- All functions are allocation-light and header-only; keep them that way.

## Verification
- `tests/test_fused_graph_emitter.cpp`, `test_fused_graph_compiler.cpp`, `test_fused_graph_activation.cpp`, `test_fused_real_nodes.cpp`, `test_fused_variant_key.cpp`, `test_fused_blend_glsl.cpp`.
- Activation plan: tracked in `DEV_LOG/plans/`.

## Child DOX Index
None — leaf subsystem.
