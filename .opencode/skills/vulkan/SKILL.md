---
name: vulkan
description: Use when adding, modifying, or debugging Vulkan compute work in this procedural texture/node-graph engine — new node types backed by compute shaders, memory allocation, changes to descriptor set layouts or push constants, dispatch/workgroup sizing, commands, command buffers, synchronization barriers between chained passes, or SPIR-V shader changes.
compatibility: opencode
---

## What I do

Compute pipelines, descriptor set layouts, push constants, workgroup dispatch, synchronization barriers, SPIR-V shaders, validation layer debugging, image format/precision changes.

## SDK lookup

| Step | Command | Fallback |
|---|---|---|
| **Find SDK** | `echo $env:VULKAN_SDK` | `Get-ChildItem C:\VulkanSDK\` |
| **List headers** | `Get-ChildItem "$env:VULKAN_SDK\Include\vulkan"` | — |
| **Search for symbol** | `grep "symbol" C:\VulkanSDK\<ver>\Include\vulkan` | `Select-String` (PowerShell) or `findstr` (cmd) |

**Prefer the `grep` tool** over per-file shell commands — it searches all headers at once.

Key headers: `vulkan_core.h` (core), `vulkan_beta.h` (extensions), `vulkan.hpp` (C++ wrapper).

## Looking up a function

1. `grep "vkFunctionName" C:\VulkanSDK\<ver>\Include\vulkan` — find the declaration line
2. `read` the file at that line — look for `VKAPI_ATTR void VKAPI_CALL` (not the typedef)
3. `grep "vkFunctionName" src/` — find how TextureSynth calls it
4. `read` usage in context (typically `EngineDispatch.cpp` for dispatch, `EnginePassCompile.cpp` for pipeline setup)

## Verification

- [ ] Resource ownership/lifetime correct
- [ ] Boundary conditions handled, not just common case
- [ ] Follows existing engine patterns (no second way to do the same thing)
- [ ] Changes don't break dependent resources/layouts/contracts
- [ ] Behavior preserved unless change was the explicit goal
