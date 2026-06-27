#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/register_allocation/GraphColorer.hpp"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace te {

struct FusedResult {
    std::string source;
    std::string error;
    uint32_t                external_inputs = 0;     // total count (for runtime slot limits)
    std::vector<uint32_t>   external_socket_masks;   // per-node bitmask for cache key
    std::vector<uint32_t>   internal_producer_indices; // flat per-socket local_index for cache key

    [[nodiscard]] constexpr bool ok() const noexcept { return error.empty(); }
};

// Emit fused GLSL for a chain of nodes.
// global_param_slots: maps each node_id to its absolute SSBO slot.
// chain_base_slot: the minimum global slot among all chain nodes (pc.param_base_slot).
// coloring: if non-null, use r[color] naming instead of _local_N.
FusedResult emit_fused_subgraph(
    const ActivePath& path,
    const GraphIR& ir,
    const NodeLibrary& lib,
    uint32_t chain_base_slot,
    const std::unordered_map<NodeId, int>& global_param_slots,
    const register_allocation::ColoringResult* coloring = nullptr,
    const std::unordered_set<ResourceUUID, ResourceUUIDHash>* active_resources = nullptr);

} // namespace te
