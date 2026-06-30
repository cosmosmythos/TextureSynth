# memory_allocation — VRAM Image Aliasing & Format-Aware Coloring

## Purpose
Format-aware interval-graph coloring for VkImage memory sharing between chains. Maps `ResourceUUID → alias color class` so resources with non-overlapping lifetimes and compatible storage formats share physical memory via `VK_IMAGE_CREATE_ALIAS_BIT`.

## Ownership
All files under `src/engine/memory_allocation/`. Namespace: `te::memory_allocation`.

| File | Role |
|---|---|
| `AliasColorer.hpp/cpp` | Format-aware interval coloring: lifetimes → color classes. |

## Local Contracts
- **Format compatibility**: two resources can alias only if `storage_format_bytes(a) == storage_format_bytes(b)`. A Mono@F32 (4 bytes/pixel) never shares with RGBA@F32 (16 bytes/pixel).
- **Color 0 = pinned**: final output, single-pass resources, and UINT32_MAX lifetimes are excluded from aliasing.
- **Input**: `ComputePass` vector + `GraphIR` + `NodeLibrary` (for format resolution). Or pre-computed lifetimes + formats for testing. Lifetimes passed to `compute_from_lifetimes()` MUST be extended for cross-chain consumers (see `FusedGraphCompiler.cpp` step 8) — raw pass-level lifetimes are insufficient because a chain dispatch reads all cross-chain inputs simultaneously.
- **Output**: `AliasColoringResult` with `color_classes` map, `lifetimes` map, and `groups_created` count.
- **Consumed by**: `ResourceManager::allocate_for_graph()` uses `color_classes` to decide which VkImages share memory.

## Relationship to register_allocation
- `register_allocation::GraphColorer` maps `ResourceUUID → GLSL register` (within a chain/dispatch).
- `memory_allocation::AliasColorer` maps `ResourceUUID → alias group` (between chains/dispatches).
- They solve different problems with the same algorithm (interval graph coloring).

## Work Guidance
- The `StorageFormat` struct (channels × depth) determines bytes/pixel. Use `storage_format_bytes()` for the comparison.
- Resources with `last_pass == UINT32_MAX` are pinned (never aliasable) — the final output falls in this category.
- Resources with `first_pass == last_pass` (single-pass lifetime) don't need aliasing — they can live in GPU registers.

## Verification
- `tests/test_aliasing.cpp` — integration tests for alias allocation.
