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

// Per-heap memory breakdown. Matches a single VkMemoryHeap in VkPhysicalDeviceMemoryProperties::memoryHeaps[].
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

    // Diagnostic: aliasing ratio within this heap. 1.0 = exact; <1.0 = aliasing (Stage 6); >1.0 = fragmentation/alignment waste.
    double aliasing_efficiency() const {
        return vma_allocation_bytes == 0
                 ? 1.0
                 : double(usage_bytes) / double(vma_allocation_bytes);
    }
};

// Telemetry: combines logical (live_/retired_ byte sums) with VMA's physical view (block bytes, allocation bytes, per-heap breakdown). Used by Python for VRAM observation.
struct VmaStatsReport {
    // Logical view
    size_t  node_resource_count    = 0;   // live_.size()
    size_t  node_resource_bytes    = 0;   // sum of live image byte sizes
    size_t  retired_count          = 0;   // retired_.size()
    size_t  retired_bytes          = 0;   // sum of retired image byte sizes

    // VMA totals (all heaps rolled up)
    size_t  vma_block_bytes        = 0;   // VmaStats::blockBytes
    size_t  vma_allocation_bytes   = 0;   // VmaStats::allocationBytes
    size_t  vma_unused_range_bytes = 0;   // blockBytes - allocationBytes

    // Device-local-only rollup (excludes host-visible staging buffers).
    // Aliasing only applies to device-local images, so this is the
    // physically meaningful number for an "aliasing efficiency" check.
    size_t  device_local_allocation_bytes = 0;

    // Configured soft limit (not GPU budget); logs warning when exceeded.
    size_t  warning_threshold_bytes = 0;

    // Real GPU memory (sum of device-local heaps only). What an artist-facing VRAM panel would display.
    size_t  gpu_budget_bytes = 0;          // sum of device-local heap budgets
    size_t  gpu_usage_bytes  = 0;          // sum of device-local heap usages
    float   gpu_pressure     = 0.0f;       // gpu_usage / gpu_budget, clamped 0..1

    // Per-heap breakdown. Size == device's memoryHeapCount (including non-device-local heaps).
    std::vector<VmaHeapStats> heap_stats;

    // Aliasing ratio over device-local memory only. 1.0 = perfect; <1.0
    // = aliasing is saving VRAM (Stage 6); >1.0 = fragmentation/overhead.
    // Excludes host-visible staging buffers which never alias.
    double aliasing_efficiency() const {
        return node_resource_bytes == 0
                 ? 1.0
                 : double(device_local_allocation_bytes) / double(node_resource_bytes);
    }
};

class ResourceManager {
public:
    // Soft limit, MB. Default 1 GB. allocate_for_graph() returns false when exceeded.
    void set_memory_budget_mb(size_t mb) { budget_bytes_ = mb * 1024ull * 1024ull; }

    // Allocate one image per (node_id, output_index) in the active subgraph. Returns false if budget exceeded.
    // When color_classes is provided, resources sharing a color > 0 alias VkDeviceMemory (Stage 6 aliasing).
    bool allocate_for_graph(VulkanContext& ctx,
                            const GraphIR& ir,
                            const NodeLibrary& lib,
                            uint32_t width, uint32_t height,
                            VkFormat default_format,
                            std::string* error = nullptr,
                            const std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash>* color_classes = nullptr);

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

    // Read-only view of the live resource map. Exposed for tests. Do not mutate.
    const auto& live_resources() const { return live_; }

    // Access alias pool generation counter for staleness detection.
    const auto& alias_pools() const { return alias_pools_; }

    // Record a write to an aliased resource: increment group generation and
    // stamp the resource. Called by Engine after issuing a write-barrier.
    void record_alias_write(uint32_t group_id, NodeResource& r) {
        if (group_id == 0 || group_id > alias_pools_.size()) return;
        auto& ag = alias_pools_[group_id - 1];
        ag.current_gen++;
        r.alias_gen = ag.current_gen;
    }

    // Snapshot of current memory state. Requires VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT for real gpu_*/heap_stats budget/usage.
    VmaStatsReport get_vma_stats(VulkanContext& ctx) const;

private:
    struct Retired {
        NodeResource res;
        uint32_t frames_remaining;
    };

    size_t pixel_bytes_(VkFormat f) const; // returns bytes per pixel
    bool create_image_(VulkanContext& ctx, NodeResource& r,
                       uint32_t w, uint32_t h, const std::string& dbg,
                       VmaPool pool = nullptr, bool can_alias = false);
    void destroy_image_(VulkanContext& ctx, NodeResource& r);

    struct AliasGroup {
        uint32_t   color       = 0;       // color class id from PassPlan::color_classes
        VkFormat   format      = VK_FORMAT_UNDEFINED;
        VkExtent3D extent      = {0, 0, 1};
        // Primary = the NodeResource whose VmaAllocation backs every image in
        // this group. Stored so destruction walks siblings-then-primary in the
        // right order (vmaDestroyImage on the primary frees the shared alloc).
        ResourceUUID primary{};
        uint64_t   current_gen = 1;       // incremented each time any member is written
    };
    std::vector<AliasGroup> alias_pools_;

    std::unordered_map<ResourceUUID, NodeResource, ResourceUUIDHash> live_;
    std::vector<Retired> retired_;
    size_t current_bytes_ = 0;
    size_t budget_bytes_  = 1024ull * 1024ull * 1024ull; // 1 GB default
};

} // namespace te
