# src/engine — Vulkan Compute Engine Core

## Purpose
The static `engine` library. Owns the full GPU pipeline: Vulkan instance/device, node library, graph → IR → pass plan compilation, resource/bindless-slot management, async readback, and dispatch. Consumed by the `texturesynth_core` nanobind binding (addon) and the optional `viewer` exe.

## Ownership
- All files under `src/engine/` except `src/engine/graphfusion/` and `src/engine/register_allocation/` (which have their own AGENTS.md files).
- Linked as `target_link_libraries(... PUBLIC engine)` — public include root is `src/`.

## Local Contracts
- **Namespace**: `te` (do not introduce a second top-level namespace).
- **One VkInstance per process.** Only `VulkanContext` creates the instance. Call `Engine::shutdown()` before constructing a new `Engine` (root §5 gotcha).
- **Stage flow (strictly one direction):**
  `Graph` (`Graph.hpp`) → `GraphIR` (`GraphIR.hpp`) → `PassPlan` (`PassPlan.hpp`) → `PassExec`/`ChainExec` (`Engine.hpp`) → `record_dispatch` (`EngineDispatch.cpp`) → `AsyncReadback`.
  Never let a later stage reach back into an earlier struct's invariants.
- **Include cycle rule** (documented in `Graph.hpp:67-71`): `PassKind` lives in `Graph.hpp` so `NodeType` can use it without pulling `PassPlan.hpp`. Preserve this layering.
- **Thread entry contract**: every public mutating call takes `entry_mutex()` first (the `TE_GUARD_READY` macro in `Engine.hpp:37` or the manual `check_engine_ready` lock in `bindings.cpp`). The async compile path runs on `std::future` workers; results land in `pending_passes_` and are installed only on the main thread via `poll_pending_compiles()`.
- **Bindless slots**: `BindlessTable` is the single source of truth for `res_sampled_slot_` / `res_storage_slot_`. Engine `PARAM_RING` must equal `BindlessTable::PARAM_RING_SIZE` (static_assert in `Engine.hpp:317`).
- **Error model**: `set_error_()` writes an `EngineError{code, message, phase, failed_node, generation}` record. Bindings translate this to Python `RuntimeError`. Never throw across the Vulkan boundary.
- **Retirement**: passes and images go through `retired_passes_` / `retired_images_` with `MAX_FRAMES_IN_FLIGHT + 2` countdown before destruction (GPU may still be reading them).

## Work Guidance
- C++20, MSVC `/W4 /permissive- /Zc:preprocessor`. No `\u2014` em-dashes in string literals (root §5).
- Shader variants: a node's compile-time specialization comes from `ShaderVariantKey` (built from `NodeType::variant_flags`). See `EnginePassCompile.cpp`.
- `MAX_NODE_PARAMS = 8192` (`Graph.hpp:44`) is the shared SSBO upper bound — bounds-check in `GraphCompiler` before extending.
- When adding a node input/output format, update `channel_to_vk_format()` in `Graph.hpp:49` AND the format post-process in the compiler.
- **Storage format system (Substance Designer model)**: format = `ChannelFormat` (Mono/UV/RGB/RGBA) x `BitDepth` (F8/F16/F32), composed via `StorageFormat{channels, depth}`. Three resolver functions in `StorageFormat.cpp`: `storage_format_to_vk()`, `storage_format_bytes()`, `storage_format_glsl_qualifier()`. The shader layout qualifier and VkImage allocation MUST both come from the same `StorageFormat` -- never hardcode `"rgba32f"` etc. The old `channel_to_vk_format(ChannelFormat)` is a deprecated F16 shim; new code passes `StorageFormat`. Per-node depth is resolved by `resolve_node_depths(ir)` via 3-mode inheritance (Auto/MatchInput/Absolute) and stamped onto `ValidatedNode::resolved_depth` before fusion compiles. Single dummy image (`Engine::dummy_image_`, 1x1 RGBA32F) serves all formats (Vulkan sampled-image reads auto-convert).

## Verification
- C++ unit tests: `tests/test_context.cpp`, `test_full_pipeline.cpp`, `test_graph_validation.cpp`, `test_async_readback.cpp`, `test_aliasing.cpp`, `test_dirty_set.cpp`, `test_image_upload.cpp`, `test_mask_node.cpp`, `test_mute_middle_node.cpp`, `test_timestamps.cpp`, `test_combine_rgba.cpp`, `test_noise_nodes.cpp`.
- Build: `cmake -S . -B build -DBUILD_TESTS:BOOL=ON` then `cmake --build build --config Release --target engine_tests` (never `--target clean`).

## Child DOX Index
- [`graphfusion/AGENTS.md`](graphfusion/AGENTS.md) — chain fusion (DAG → planner → emitter → fused compiler). Distinct compilation model from the per-pass path above.
- [`register_allocation/AGENTS.md`](register_allocation/AGENTS.md) — graph-coloring register allocator for fused shader chains. Chaitin-Briggs + linear scan; interval graph perfection guarantees optimal greedy coloring.
