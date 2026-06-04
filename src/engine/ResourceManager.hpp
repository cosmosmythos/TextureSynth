#pragma once
#include "engine/NodeResource.hpp"
#include "engine/GraphIR.hpp"
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace te {

class VulkanContext;
class NodeLibrary;

// Per-heap memory breakdown. Matches a single VkMemoryHeap in
// VkPhysicalDeviceMemoryProperties::memoryHeaps[].
//
// `is_device_local` reflects the heap's VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
// flag -- the *heap* flag, not the per-type flags. On a discrete GPU the
// device-local heap is your fast VRAM (e.g. 8 GB on a GeForce 3060). On
// an integrated/UMA GPU the single heap has both DEVICE_LOCAL *and*
// HOST_VISIBLE types, and isDeviceLocal is still true (the heap itself
// is the "fast" pool the GPU prefers).
//
// `budget_bytes` and `usage_bytes` are the OS-level residency numbers
// from VkPhysicalDeviceMemoryBudgetPropertiesEXT, populated by VMA when
// the allocator is created with VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT
// (see VulkanContext.cpp). They reflect what the *driver* thinks, not
// what VMA sub-allocated. They include swapchain images, pipelines,
// descriptor pools, and any VkDeviceMemory the engine or other
// components allocated outside VMA.
//
// `vma_allocation_bytes` and `vma_block_bytes` are VMA's *internal*
// accounting for this heap: how many bytes VMA has handed to the user
// (allocationBytes) and how many bytes Vulkan-allocated (blockBytes).
// The difference is VMA's free/fragmentation inside its own pool.
//
// `pressure` is usage_bytes / budget_bytes, clamped 0..1. A useful
// "VRAM gauge" for an artist UI.
struct VmaHeapStats {
    uint32_t    index                  = 0;     // VkMemoryHeap index
    bool        is_device_local        = false; // VK_MEMORY_HEAP_DEVICE_LOCAL_BIT
    const char* label                  = "unknown"; // human-friendly name
    size_t      heap_size_bytes        = 0;     // VkMemoryHeap::size (raw)
    size_t      budget_bytes           = 0;     // OS-level VRAM budget
    size_t      usage_bytes            = 0;     // OS-level VRAM usage
    size_t      vma_allocation_bytes   = 0;     // VMA's sub-alloc bytes in this heap
    size_t      vma_block_bytes        = 0;     // VMA's VkDeviceMemory in this heap
    uint32_t    vma_block_count        = 0;
    uint32_t    vma_allocation_count   = 0;
    float       pressure               = 0.0f;  // usage / budget, clamped 0..1

    // Diagnostic: aliasing ratio within *this* heap.
    // == 1.0 means VMA hands back exactly what we asked for. < 1.0
    // means aliasing (one VmaAllocation backs multiple NodeResources,
    // Stage 6 transient aliasing). > 1.0 means fragmentation or
    // alignment waste in this heap.
    double aliasing_efficiency() const {
        return vma_allocation_bytes == 0
                 ? 1.0
                 : double(usage_bytes) / double(vma_allocation_bytes);
    }
};

// Telemetry: combines the ResourceManager's *logical* view (live_ +
// retired_ byte sums) with VMA's *physical* view (block bytes,
// allocation bytes, per-heap breakdown). Used by Python to observe how
// much VRAM the engine actually consumes and how well transient aliasing
// (Stage 6) is paying off.
//
// Field naming convention:
//   - `*_bytes` suffixes are always in bytes (raw VkDeviceSize cast to
//     size_t; never formatted with KB/MB at this layer).
//   - `gpu_*` fields are the *real GPU* numbers (OS-level residency,
//     not a configured threshold).
//   - `warning_threshold_bytes` is the configured soft limit from
//     set_memory_budget_mb(). The original name `budget_bytes` was
//     misleading -- it suggested GPU VRAM but was a config knob. Now
//     clearly named.
struct VmaStatsReport {
    // Logical view (was: original v1 fields, semantics preserved)
    size_t  node_resource_count    = 0;   // live_.size()
    size_t  node_resource_bytes    = 0;   // sum of live image byte sizes
    size_t  retired_count          = 0;   // retired_.size()
    size_t  retired_bytes          = 0;   // sum of retired image byte sizes

    // VMA totals (all heaps rolled up)
    size_t  vma_block_bytes        = 0;   // VmaStats::blockBytes
    size_t  vma_allocation_bytes   = 0;   // VmaStats::allocationBytes
    size_t  vma_unused_range_bytes = 0;   // blockBytes - allocationBytes

    // Configured soft limit. Not a GPU number -- just a threshold the
    // engine logs a warning against when total logical bytes would
    // exceed it. Replaces the misleading v1 `budget_bytes` field.
    size_t  warning_threshold_bytes = 0;

    // Real GPU memory (sum of device-local heaps only). What an
    // artist-facing "VRAM usage" panel would display. Zero on a system
    // with no device-local heaps (which is allowed by spec but never
    // happens in practice -- the spec guarantees at least one
    // DEVICE_LOCAL heap).
    size_t  gpu_budget_bytes = 0;          // sum of device-local heap budgets
    size_t  gpu_usage_bytes  = 0;          // sum of device-local heap usages
    float   gpu_pressure     = 0.0f;       // gpu_usage / gpu_budget, clamped 0..1

    // Per-heap breakdown. Always populated (size == device's
    // memoryHeapCount, including non-device-local heaps).
    std::vector<VmaHeapStats> heap_stats;

    // Global aliasing ratio: VMA's reported physical allocation bytes
    // (all heaps) divided by our logical resource bytes. == 1.0 means
    // perfect packing. < 1.0 means aliasing is saving memory (Stage 6
    // transient aliasing reuses one VmaAllocation across multiple
    // NodeResources). > 1.0 means fragmentation or sub-allocation
    // overhead.
    double aliasing_efficiency() const {
        return node_resource_bytes == 0
                 ? 1.0
                 : double(vma_allocation_bytes) / double(node_resource_bytes);
    }
};

class ResourceManager {
public:
    // Soft limit, MB. Default 1 GB. When the engine would allocate
    // more than this, allocate_for_graph() returns false and logs
    // "Memory Budget Exceeded". This is a *warning threshold*, not a
    // GPU fact -- use VmaStatsReport::gpu_budget_bytes for that.
    void set_memory_budget_mb(size_t mb) { budget_bytes_ = mb * 1024ull * 1024ull; }

    // Allocate one image per (node_id, output_index=0) in the active subgraph.
    // Returns false if budget would be exceeded; populates `error`.
    bool allocate_for_graph(VulkanContext& ctx,
                            const GraphIR& ir,
                            const NodeLibrary& lib,
                            uint32_t width, uint32_t height,
                            VkFormat default_format,
                            std::string* error = nullptr);

    // Defer destruction; caller must call tick() each frame.
    void retire_all(VulkanContext& ctx);
    void tick(VulkanContext& ctx);
    void shutdown(VulkanContext& ctx); // destroys everything immediately

    const NodeResource* get(ResourceUUID id) const {
        auto it = live_.find(id);
        return it == live_.end() ? nullptr : &it->second;
    }
    NodeResource* get(ResourceUUID id) {
        auto it = live_.find(id);
        return it == live_.end() ? nullptr : &it->second;
    }

    size_t current_bytes() const { return current_bytes_; }
    size_t budget_bytes() const  { return budget_bytes_; }  // configured warning threshold
    size_t live_count() const { return live_.size(); }

    // Read-only view of the live resource map. Exposed for tests that
    // need to introspect VmaAllocation names (via vmaGetAllocationInfo)
    // or other per-resource state. Do not mutate -- the manager owns
    // the lifetimes (retire_all / tick / shutdown).
    const auto& live_resources() const { return live_; }

    // Snapshot of the current memory state. Safe to call any time after
    // allocate_for_graph(); vmaCalculateStatistics is O(blocks) and
    // vmaGetHeapBudgets is O(heaps), both fast. The VulkanContext is
    // needed to reach the VmaAllocator. Requires the engine to be
    // initialized with VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT
    // (VulkanContext.cpp does this) for the gpu_* and heap_stats[*].
    // budget/usage fields to be real numbers; without it, VMA fills
    // them with 80%-of-heap-size estimates.
    VmaStatsReport get_vma_stats(VulkanContext& ctx) const;

private:
    struct Retired {
        NodeResource res;
        uint32_t frames_remaining;
    };

    size_t pixel_bytes_(VkFormat f) const; // returns bytes per pixel
    bool create_image_(VulkanContext& ctx, NodeResource& r,
                       uint32_t w, uint32_t h, const std::string& dbg);
    void destroy_image_(VulkanContext& ctx, NodeResource& r);

    std::unordered_map<ResourceUUID, NodeResource, ResourceUUIDHash> live_;
    std::vector<Retired> retired_;
    size_t current_bytes_ = 0;
    size_t budget_bytes_  = 1024ull * 1024ull * 1024ull; // 1 GB default
};

} // namespace te
