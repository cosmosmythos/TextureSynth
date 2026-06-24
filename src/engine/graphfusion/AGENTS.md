# src/engine/graphfusion — Chain Fusion Subsystem

## Purpose
Fuses a chain of compute nodes into a single shader so a subgraph runs in one dispatch instead of N. Replaces the per-pass dispatch path (`PassExec`) for eligible chains while remaining bit-identical to the unfused reference per node.

## Ownership
All of `src/engine/graphfusion/`:
- `DAG.hpp` — directed-acyclic-graph + topological primitives (`te::dag`).
- `ActivePathTracer.{hpp,cpp}` — walks from the active/output node back to sources, yielding the live path.
- `FusionPlanner.hpp` — groups path nodes into `FusionGroup`s bounded by register budget.
- `RegisterAllocator.hpp` — estimates/allocates live vec4 registers (`te::reg`).
- `GlslBuilder.hpp` — emits GLSL statement strings from fused nodes.
- `FusedGraphEmitter.{hpp,cpp}` — assembles the fused compute shader source.
- `FusedGraphCompiler.{hpp,cpp}` — entry point: `compile(GraphIR, NodeLibrary, active_node_id)` → `CompileGraphResult`.

## Local Contracts
- **Namespace**: `te::fusion` for planner/reg/dag helpers; `te::FusedGraphCompiler` for the entry class.
- **Bit-identity contract**: fused output must equal the unfused per-pass output for each node. Tests in `tests/test_fused_*.cpp` enforce this; do not land a fusion change that breaks `test_fused_real_nodes.cpp`.
- **Register budget**: `RegisterAllocator::DEFAULT_BUDGET`; `FusedGraphCompiler::DEFAULT_REG_BUDGET = 48`. If a planned group exceeds budget, `FusionPlan::needs_split` becomes true and the planner splits the path. Splitting enforces the consumer constraint: no intermediate node in a group may have a successor in the active path outside that group. This ensures all intermediate values consumed by downstream passes/groups are written to VRAM.
- **DAG topology (root §5 gotcha)**: the planner's DAG MUST be built from REAL `ir.connections` edges, not synthetic `path[i-1] -> path[i]` adjacency. Linear adjacency hides fan-out/merge and lets `split_path`'s consumer-constraint check pass on invalid groups. Also `FusionPlanner::is_valid_path` accepts any topological order (siblings/fan-out allowed), not just linear chains — `ActivePathTracer` returns a topo sort of a sub-DAG, which can include parallel branches feeding a merge node.
- **active_resources invariant**: every node whose output is consumed by a pass in a DIFFERENT chain (cross-group producer) MUST appear in `plan.active_resources`, or `ResourceManager` will skip its image allocation and the downstream chain's `texelFetch` reads garbage. This includes intermediate (non-tail) chain nodes — they feed in-chain consumers via registers, but cross-chain consumers via VRAM textures. The compiler's cross-group edge scan enforces this.
- **Inputs**: only `GraphIR` + `NodeLibrary` + `active_node_id`. Do not reach into `Engine` or `PassPlan` from here — fusion is a pure transform that the engine *consumes*.
- **Active path**: only nodes on the `ActivePathTracer` result are candidates for fusion. Off-path nodes (bake targets, dead branches) are never fused.
- **GLSL contract**: emitted code calls node functions following the `shader_assets/nodes/README.md` signature contract (`vec4 node_<name>(vec2 uv, ...)`). No `sampler2D` / `imageLoad` inside emitted node calls.
- **FusedVariantKey contract**: the cache key must include everything that affects generated GLSL. Currently: `node_type_ids`, `param_socket_masks`, `input_counts`, `feature_flags`, `external_socket_masks`, **`internal_producer_indices`**, `epoch=8`. `internal_producer_indices` is the per-socket local_index of each in-chain producer (flat, node-major, `UINT32_MAX` for external/const) — two graphs with the same node types but swapped internal A/B wiring (e.g. `Blend(Value→a, Simplex→b)` vs `Blend(Simplex→a, Value→b)`) emit different GLSL but collide on every older field, so they MUST also differ here. If you add a new GLSL-affecting field: edit `FusedVariantKey` (decl + `operator==` + `hash()`), populate it in `build_fused_key()` AND in the emitter→`Chain`→key propagation path (`FusedGraphEmitter.cpp` → `FusedResult` → `FusedGraphCompiler.cpp` → `Chain` → `build_fused_key`), serialize/deserialize it in `ShaderCache::write_sidecar_()`/`sidecar_matches_()`, then bump `epoch`. Skipping any of these sites is how the swapped-wiring collision shipped. Residual gap: unconnected-socket `ConstSrc{value}` defaults are baked into GLSL as `vec4(literal)` but not yet in the key — a node-library default change on an unconnected socket could still collide; flagged for future work.

## Work Guidance
- When adding a node that should fuse: confirm `NodeType::pass_kind == Compute` and that its GLSL is pure (no side effects, no texture fetches inside the node fn). Image nodes (`Sampler2D` input) ARE fusible — the chain slot resolution reads live slots at dispatch time, so late async uploads work correctly within fused chains.
- Debug fused vs. unfused divergence with `tests/compare_fusion_outputs.bat` and `test_fused_real_nodes.cpp`.
- `FusionPlanner` is header-only and templated on `NodeId`; keep it allocation-light since it runs at compile time on the live path.

## Verification
- `tests/test_active_path_tracer.cpp`, `test_fused_graph_emitter.cpp`, `test_fused_graph_compiler.cpp`, `test_fused_graph_activation.cpp`, `test_fused_real_nodes.cpp`, `test_fused_variant_key.cpp`.
- Activation plan: `.opencode/plans/graph-fusion-activation.md` tracks rollout status.

## Child DOX Index
None — leaf subsystem.
