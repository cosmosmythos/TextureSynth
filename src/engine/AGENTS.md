# src/engine — Vulkan Compute Engine Core

## Purpose
The static `engine` library. Owns the full GPU pipeline: Vulkan instance/device, node library, graph → IR → pass plan compilation, resource/bindless-slot management, async readback, and dispatch. Consumed by the `texturesynth_core` nanobind binding (addon) and the optional `viewer` exe.

## Ownership
- All files under `src/engine/` except `src/engine/graphfusion/`, `src/engine/register_allocation/`, and `src/engine/memory_allocation/` (which have their own AGENTS.md files).
- Linked as `target_link_libraries(... PUBLIC engine)` — public include root is `src/`.

## Local Contracts
- **Namespace**: `te` (do not introduce a second top-level namespace).
- **One VkInstance per process.** Only `VulkanContext` creates the instance. Call `Engine::shutdown()` before constructing a new `Engine` (root §5 gotcha).
- **Stage flow (strictly one direction):**
  `Graph` (`Graph.hpp`) → `GraphIR` (`GraphIR.hpp`) → `PassPlan` (`PassPlan.hpp`) → `PassExec`/`ChainExec` (`Engine.hpp`) → `record_dispatch` (`EngineDispatch.cpp`) → `AsyncReadback`.
  Never let a later stage reach back into an earlier struct's invariants.
- **Include cycle rule** (documented in `Graph.hpp:102-107`): `PassKind` lives in `Graph.hpp` so `NodeType` can use it without pulling `PassPlan.hpp`. Preserve this layering.
- **Thread entry contract**: every public mutating call takes `entry_mutex()` first (the `TE_GUARD_READY` macro in `Engine.hpp:37` or the manual `check_engine_ready` lock in `bindings.cpp`). The async compile path runs on `std::future` workers; results land in `pending_passes_` and are installed only on the main thread via `poll_pending_compiles()`.
- **Bindless slots**: `BindlessTable` is the single source of truth for `res_sampled_slot_` / `res_storage_slot_`. Engine `PARAM_RING` must equal `BindlessTable::PARAM_RING_SIZE` (static_assert in `Engine.hpp:364`).
- **Chain slot resolution (live, not snapshot)**: `ChainExec::slot_sources[]` maps each external slot to `(member_pass_index, input_index)`. At dispatch time (`EngineDispatch.cpp:92, 139`), `record_chain_dispatch_()` reads `passes_[member_pass_idx].in_sampled_slots[input_index]` live — NOT the snapshotted `chain_in_sampled_slots[]`. This prevents stale dummy slots when async image upload completes after `populate_chains_()`. The snapshot (`chain_in_sampled_slots[]`) is retained as a fallback for bypassed chains.
- **Error model**: `set_error_()` writes an `EngineError{code, message, phase, failed_node, generation}` record. Bindings translate this to Python `RuntimeError`. Never throw across the Vulkan boundary.
- **Retirement**: passes and images go through `retired_passes_` / `retired_images_` with `MAX_FRAMES_IN_FLIGHT + 2` countdown before destruction (GPU may still be reading them).

## Work Guidance
- C++20, MSVC `/W4 /permissive- /Zc:preprocessor`. No `\u2014` em-dashes in string literals (root §5).
- Shader variants: a node's compile-time specialization comes from `ShaderVariantKey` (built from `NodeType::variant_flags`). See `EnginePassCompile.cpp`.
- `MAX_NODE_PARAMS = 8192` (`Graph.hpp:44`) is the shared SSBO upper bound — bounds-check in `GraphCompiler` before extending.
- When adding a node input/output format, update the format resolution in the compiler and `StorageFormat.cpp`.
- **Storage format system (Substance Designer model)**: format = `ChannelFormat` (Mono/UV/RGB/RGBA) x `BitDepth` (F8/F16/F32), composed via `StorageFormat{channels, depth}`. Three resolver functions in `StorageFormat.cpp`: `storage_format_to_vk()`, `storage_format_bytes()`, `storage_format_glsl_qualifier()`. The shader layout qualifier and VkImage allocation MUST both come from the same `StorageFormat` -- never hardcode `"rgba32f"` etc. Per-node depth is resolved by `resolve_node_depths(ir)` via 3-mode inheritance (Auto/MatchInput/Absolute) and stamped onto `ValidatedNode::resolved_depth` before fusion compiles. Single dummy image (`Engine::dummy_image_`, 1x1 RGBA32F) serves all formats (Vulkan sampled-image reads auto-convert).
- **Multi-pass architecture**: nodes with `pass_count > 1` (e.g. separable blur) are singleton chains with sub-pass GLSL. `Chain::sub_pass_count` / `sub_pass_glsl` / `sub_pass_variant_keys` hold per-sub-pass data. `ChainExec::sub_pipelines` / `Intermediate` / `sub_out_storage_slots` hold runtime state. `populate_chains_()` allocates intermediate images (`Image::create(ctx_, w, h)`) with bindless sampled+storage slots, compiles sub-pipelines with `VkSpecializationInfo` (specialization constant = sub-pass index). `record_chain_dispatch_()` loops over sub-passes with `barrier_compute_write_to_sampled` between passes. `ts_pass_index` (GLSL `layout(constant_id = 0)`) is baked into SPIR-V for dead-code elimination. `pc.pass_index` is set to sub-pass index at dispatch time. Intermediate images are retired via `retired_images_` with frame counter.

## Verification
- C++ unit tests: all `tests/test_*.cpp` files (30 files covering context, pipeline, validation, fusion, blend, blur, storage format, depth resolution, etc.). See `tests/AGENTS.md` for suite contracts.
- Build: `cmake -S . -B build -DBUILD_TESTS:BOOL=ON` then `cmake --build build --config Release --target engine_tests` (never `--target clean`).

## Child DOX Index
- [`graphfusion/AGENTS.md`](graphfusion/AGENTS.md) — chain fusion (DAG → planner → emitter → fused compiler). Distinct compilation model from the per-pass path above.
- [`register_allocation/AGENTS.md`](register_allocation/AGENTS.md) — graph-coloring register allocator for fused shader chains. Chaitin-Briggs + linear scan; interval graph perfection guarantees optimal greedy coloring.
- [`memory_allocation/AGENTS.md`](memory_allocation/AGENTS.md) — format-aware interval coloring for VkImage memory sharing (aliasing) between chains.
