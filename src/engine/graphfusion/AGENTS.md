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
- **Register budget**: `RegisterAllocator::DEFAULT_BUDGET`; `FusedGraphCompiler::DEFAULT_REG_BUDGET = 48`. If a planned group exceeds budget, `FusionPlan::needs_split` becomes true and the emitter splits at `FusionGroup::split_point`.
- **Inputs**: only `GraphIR` + `NodeLibrary` + `active_node_id`. Do not reach into `Engine` or `PassPlan` from here — fusion is a pure transform that the engine *consumes*.
- **Active path**: only nodes on the `ActivePathTracer` result are candidates for fusion. Off-path nodes (bake targets, dead branches) are never fused.
- **GLSL contract**: emitted code calls node functions following the `shader_assets/nodes/README.md` signature contract (`vec4 node_<name>(vec2 uv, ...)`). No `sampler2D` / `imageLoad` inside emitted node calls.

## Work Guidance
- When adding a node that should fuse: confirm `NodeType::pass_kind == Compute` and that its GLSL is pure (no side effects, no texture fetches inside the node fn). Read-only samplers like `image` are NOT fusion candidates.
- Debug fused vs. unfused divergence with `tests/compare_fusion_outputs.bat` and `test_fused_real_nodes.cpp`.
- `FusionPlanner` is header-only and templated on `NodeId`; keep it allocation-light since it runs at compile time on the live path.

## Verification
- `tests/test_active_path_tracer.cpp`, `test_fused_graph_emitter.cpp`, `test_fused_graph_compiler.cpp`, `test_fused_graph_activation.cpp`, `test_fused_real_nodes.cpp`, `test_fused_variant_key.cpp`.
- Activation plan: `.opencode/plans/graph-fusion-activation.md` tracks rollout status.

## Child DOX Index
None — leaf subsystem.
