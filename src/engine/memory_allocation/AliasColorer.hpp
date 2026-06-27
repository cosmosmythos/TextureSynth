#pragma once

#include "engine/Graph.hpp"
#include "engine/NodeResource.hpp"
#include "engine/PassPlan.hpp"
#include "engine/NodeLibrary.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace te::memory_allocation {

// Format-compatible alias group: resources in the same group must have
// the same StorageFormat (same bytes/pixel) to avoid memory waste.
// Two resources alias iff they have (a) compatible format AND (b) non-overlapping
// pass-level lifetimes.

struct AliasLifetime {
    uint32_t first_pass = UINT32_MAX;
    uint32_t last_pass  = 0;
};

// Result of alias coloring: maps each ResourceUUID to a color class.
// Color 0 = pinned (not aliasable). Colors 1,2,... = alias groups.
// Resources in the same color group share VkImage memory via VK_IMAGE_CREATE_ALIAS_BIT.
struct AliasColoringResult {
    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> color_classes;
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    uint32_t groups_created = 0;
};

// Format-aware interval-graph coloring for VRAM image aliasing.
//
// Unlike register_allocation::GraphColorer (which maps ResourceUUID → GLSL
// register within a single dispatch), this maps ResourceUUID → alias color
// class across dispatches. Resources with non-overlapping lifetimes AND
// compatible StorageFormat share a color class, enabling VkImage memory
// sharing via VK_IMAGE_CREATE_ALIAS_BIT.
//
// Algorithm:
//   1. Compute pass-level lifetimes (first_pass, last_pass) per resource.
//   2. Filter: skip final output, single-pass resources, UINT32_MAX lifetimes.
//   3. Group remaining resources by StorageFormat (same bytes/pixel).
//   4. Within each format group, run greedy first-fit interval coloring.
//   5. Assign color classes starting from 1 (0 = pinned).
//
// The format grouping ensures a Mono@F32 resource (4 bytes/pixel) never
// shares a color class with an RGBA@F32 resource (16 bytes/pixel), preventing
// memory waste from max-sized allocation.
class AliasColorer {
public:
    // Compute alias coloring for all resources in a PassPlan.
    // Requires the passes vector (for lifetime computation) and the IR+library
    // (for format resolution).
    static AliasColoringResult compute(
        const std::vector<ComputePass>& passes,
        const ResourceUUID& final_output,
        const GraphIR& ir,
        const NodeLibrary& lib);

    // Compute alias coloring from pre-computed lifetimes (for testing or
    // when lifetimes are already available).
    static AliasColoringResult compute_from_lifetimes(
        const std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash>& lifetimes,
        const std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash>& formats);

    // Check if two StorageFormats are alias-compatible (same bytes/pixel).
    static bool formats_compatible(StorageFormat a, StorageFormat b);
};

} // namespace te::memory_allocation
