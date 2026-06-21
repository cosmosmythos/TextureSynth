#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include <string>
#include <unordered_map>

namespace te {

struct FusedResult {
    std::string source;
    std::string error;
    uint32_t                external_inputs = 0;     // total count (for runtime slot limits)
    std::vector<uint32_t>   external_socket_masks;   // per-node bitmask for cache key

    [[nodiscard]] constexpr bool ok() const noexcept { return error.empty(); }
};

// Emit fused GLSL for a chain of nodes.
// global_param_slots: maps each node_id to its absolute SSBO slot.
// chain_base_slot: the minimum global slot among all chain nodes (pc.param_base_slot).
FusedResult emit_fused_subgraph(
    const ActivePath& path,
    const GraphIR& ir,
    const NodeLibrary& lib,
    uint32_t chain_base_slot,
    const std::unordered_map<NodeId, int>& global_param_slots);

} // namespace te
