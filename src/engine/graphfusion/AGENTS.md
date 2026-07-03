# src/engine/graphfusion — Chain Fusion Subsystem

## Purpose
Fuses a chain of compute nodes into a single shader so a subgraph runs in one dispatch instead of N. Replaces the per-pass dispatch path (`PassExec`) for eligible chains while remaining bit-identical to the unfused reference per node.

## Ownership
All of `src/engine/graphfusion/`:
- `DAG.hpp` — directed-acyclic-graph + topological primitives (`te::dag`).
- `FusionPlanner.hpp` — groups path nodes into `FusionGroup`s bounded by register budget.
- `RegisterAllocator.hpp` — estimates/allocates live vec4 registers (`te::reg`).
- `GlslBuilder.hpp` — emits GLSL statement strings from fused nodes.
- `FusedGraphEmitter.{hpp,cpp}` — assembles the fused compute shader source.
- `FusedGraphCompiler.{hpp,cpp}` — entry point: `compile(GraphIR, NodeLibrary, active_node_id)` → `CompileGraphResult`.

## Local Contracts
- **Namespace**: `te::dag` (DAG), `te::reg` (RegisterAllocator), `te::glsl` (GlslBuilder), `te::fusion` (FusionPlanner/FusionGroup/FusionPlan). Top-level `te` (FusedGraphEmitter, FusedGraphCompiler).
- **Bit-identity contract**: fused output must equal the unfused per-pass output for each node. Tests in `tests/test_fused_*.cpp` enforce this; do not land a fusion change that breaks `test_fused_real_nodes.cpp`.
- **Register budget**: `RegisterAllocator::DEFAULT_BUDGET`; `FusedGraphCompiler::DEFAULT_REG_BUDGET = 48`. If a planned group exceeds budget, `FusionPlan::needs_split` becomes true and the planner splits the path into register-pressure-bounded groups. Consumer-constraint enforcement has been removed from the planner; chain boundaries are now set by the unified split logic in `FusedGraphCompiler::compile()` (see below).
- **DAG topology**: the planner's DAG MUST be built from REAL `ir.connections` edges, not synthetic `path[i-1] -> path[i]` adjacency. `FusionPlanner::is_valid_path` accepts any topological order (siblings/fan-out allowed), not just linear chains — the active path is a topo sort of a sub-DAG, which can include parallel branches feeding a merge node.
- **Unified split logic**: `compute_chain_splits(path, ir, lib)` (anonymous namespace in `FusedGraphCompiler.cpp`) sets chain boundaries via two iterative rules: (1) split BEFORE any node with a `Sampler2D` input (must be in a different chain from its producer), and (2) split AFTER any cross-chain producer that is not already a tail — applies to all socket types (Sampler2D `texture()` and Vec4 `texelFetch`), not just Sampler2D. The function iterates Rule 2 until stable — adding splits can reveal new cross-chain edges.
- **Chain topo-sort**: `compile()` step 7 topologically sorts chains using Kahn's algorithm over cross-chain dependency edges from `ir.connections`. This prevents dispatching a consumer chain before its producer in diamond topologies. If a cycle is detected (sorted.size() != N), chain order is left unchanged.
- **Alias lifetime extension**: `compile()` step 8 extends cross-chain resource lifetimes before alias coloring. A chain dispatch reads all cross-chain inputs simultaneously, but pass-level lifetimes may not overlap (e.g. resource consumed at pass 4 vs pass 10, both in the same chain). Without extension, two such resources could share an alias VkImage — both descriptors point to the same image, so only the last-written data is visible. The fix extends each cross-chain resource's `last_pass` to the consuming chain's last pass index, ensuring the interval graph correctly sees them as overlapping.
- **active_resources invariant**: every node whose output is consumed by a pass in a DIFFERENT chain (cross-group producer) MUST appear in `plan.active_resources`, or `ResourceManager` will skip its image allocation and the downstream chain's `texelFetch` reads garbage. This includes intermediate (non-tail) chain nodes — they feed in-chain consumers via registers, but cross-chain consumers via VRAM textures. The compiler's cross-group edge scan enforces this.
- **Inputs**: only `GraphIR` + `NodeLibrary` + `active_node_id`. Do not reach into `Engine` or `PassPlan` from here — fusion is a pure transform that the engine *consumes*.
- **Active path**: only nodes reachable backward from the active node are candidates for fusion. Off-path nodes (bake targets, dead branches) are never fused.
- **GLSL contract**: emitted code calls node functions following the `shader_assets/nodes/README.md` signature contract (`vec4 node_<name>(vec2 uv, ...)`). No `sampler2D` / `imageLoad` inside emitted node calls.
- **FusedVariantKey contract**: the cache key must include everything that affects generated GLSL. Currently: `node_type_ids`, `param_socket_masks`, `input_counts`, `feature_flags` (3 bits format + 2 bits depth per node), `external_socket_masks`, `internal_producer_indices`, `epoch=8`. `internal_producer_indices` is the per-socket local_index of each in-chain producer (flat, node-major, `UINT32_MAX` for external/const) — two graphs with the same node types but swapped internal A/B wiring (e.g. `Blend(Value→a, Simplex→b)` vs `Blend(Simplex→a, Value→b)`) emit different GLSL but collide on every older field, so they MUST also differ here. If you add a new GLSL-affecting field: edit `FusedVariantKey` (decl + `operator==` + `hash()`), populate it in `build_fused_key()` AND in the emitter→`Chain`→key propagation path (`FusedGraphEmitter.cpp` → `FusedResult` → `FusedGraphCompiler.cpp` → `Chain` → `build_fused_key`), serialize/deserialize it in `ShaderCache::write_sidecar_()`/`sidecar_matches_()`, then bump `epoch`. Skipping any of these sites is how the swapped-wiring collision shipped. Residual gap: unconnected-socket `ConstSrc{value}` defaults are baked into GLSL as `vec4(literal)` but not yet in the key — a node-library default change on an unconnected socket could still collide; flagged for future work.
- **Multi-pass singleton chains**: nodes with `pass_count > 1` (e.g. blur) force a split BEFORE and AFTER them in the fusion compiler. They become singleton chains (`group.nodes.size() == 1`) with `sub_pass_count = type->pass_count`. The fused compiler emits per-sub-pass GLSL with different `ShaderVariantKey::specialization[0]` values. The chain's `chain.glsl` is set to empty string (per-node `pass.shader_glsl` is blanked at `FusedGraphCompiler.cpp:145`); per-sub-pass shader code lives in `chain.sub_pass_glsl[sp]`. Multi-pass nodes CANNOT fuse with other nodes — they need intermediate images between sub-passes. Without the trailing split, a multi-pass node followed by another node (e.g. blur→invert) would be grouped together, and `is_multipass` would be false (requires `group.nodes.size() == 1`), causing the node to run single-pass and produce stretched output.

## Work Guidance
- When adding a node that should fuse: confirm `NodeType::pass_kind == Compute` and that its GLSL is pure (no side effects, no texture fetches inside the node fn). Image nodes (`Sampler2D` input) ARE fusible — the chain slot resolution reads live slots at dispatch time, so late async uploads work correctly within fused chains.
- Debug fused vs. unfused divergence with `tests/compare_fusion_outputs.bat` and `test_fused_real_nodes.cpp`.
- `FusionPlanner` is header-only with methods templated on `NodeId` (the class itself is non-templated); keep it allocation-light since it runs at compile time on the live path.

## Verification
- `tests/test_fused_graph_emitter.cpp`, `test_fused_graph_compiler.cpp`, `test_fused_graph_activation.cpp`, `test_fused_real_nodes.cpp`, `test_fused_variant_key.cpp`, `test_fused_blend_glsl.cpp`.
- Activation plan: tracked in `DEV_LOG/plans/`.

## Child DOX Index
None — leaf subsystem.
