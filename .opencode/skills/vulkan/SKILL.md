---
name: vulkan
description: Use when adding, modifying, or debugging Vulkan compute work in this procedural texture/node-graph engine — new node types backed by compute shaders, memory allocation, changes to descriptor set layouts or push constants, dispatch/workgroup sizing, commands, command buffers, synchronization barriers between chained passes, or SPIR-V shader changes.
---

## What I do
- Add or modify compute pipelines and descriptor set layouts for node types
- Wire up push constants for new per-node parameters
- Size and tune workgroup dispatch for 2D image processing kernels
- Diagnose synchronization/barrier bugs between chained compute passes
- Compile and integrate new or changed SPIR-V compute shaders
- Debug validation layer warnings/errors in existing compute code

## When to use me
- Adding a new node type that requires a compute shader
- Changing a descriptor set layout (new storage image, buffer, or sampler binding)
- Modifying push constants for a node's parameters
- Adjusting workgroup/dispatch sizing for performance
- Adding or fixing synchronization barriers between chained passes in the node graph
- Debugging validation layer errors or unexpected GPU output
- Changing image format or precision (8-bit vs float/HDR) for outputs
- Modifying or adding SPIR-V compute shaders

## Locating the SDK
Try PowerShell first (most capable, present on all modern Windows):
`$env:VULKAN_SDK`

If PowerShell isn't available, try cmd:
`echo %VULKAN_SDK%`

If the variable is unset in either, find the installed version directly:
- PowerShell: `Get-ChildItem C:\VulkanSDK\`
- cmd: `dir C:\VulkanSDK\`
- Git Bash/WSL: `ls /mnt/c/VulkanSDK/` or `ls "C:\VulkanSDK"` depending on path translation

## Searching SDK contents
List the include directory first rather than guessing a filename or symbol from memory:
- PowerShell: `Get-ChildItem C:\VulkanSDK\1.4.350.0\Include\vulkan`
- cmd: `dir C:\VulkanSDK\1.4.350.0\Include\vulkan`
- Git Bash/WSL: `ls C:\VulkanSDK\1.4.350.0\Include\vulkan` (or `/mnt/c/...` path)

Once you know which header is relevant (e.g. `vulkan_core.h`, `vulkan_beta.h`), search its contents for the symbol, extension, or struct you need, since SDK versions add/deprecate things frequently:
- PowerShell: `Select-String -Path "C:\VulkanSDK\1.4.350.0\Include\vulkan\vulkan_core.h" -Pattern "VkCopyBufferInfo2"`
- cmd: `findstr "VkCopyBufferInfo2" C:\VulkanSDK\1.4.350.0\Include\vulkan\vulkan_core.h`
- Git Bash/WSL: `grep "VkCopyBufferInfo2" C:\VulkanSDK\1.4.350.0\Include\vulkan\vulkan_core.h`

## Verification checklist
Before considering Vulkan-related compute work complete, check:

- [ ] Resource ownership and lifetime are explicit and correct
- [ ] Edge cases and boundary conditions have been considered, not just the common case
- [ ] New code follows patterns already established elsewhere in the engine, rather than introducing a second way of doing the same thing
- [ ] Any deviation from existing conventions is deliberate, not accidental
- [ ] Changes don't silently break other parts of the system that depend on the same resources, layouts, or contracts
- [ ] Existing behavior is preserved unless a change in behavior was the explicit goal
- [ ] Choices are deliberate for the workload, not copy-pasted defaults