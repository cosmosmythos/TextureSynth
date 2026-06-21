#pragma once

#include "engine/Graph.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace te::register_allocation {

// A live interval represents the topological span during which a node output
// (identified by ResourceUUID) holds a value that may still be read.
struct LiveInterval {
    ResourceUUID resource;
    uint32_t start_step = 0;
    uint32_t end_step   = 0;

    [[nodiscard]] bool overlaps(const LiveInterval& other) const noexcept {
        return start_step <= other.end_step && other.start_step <= end_step;
    }

    [[nodiscard]] uint32_t length() const noexcept {
        return end_step >= start_step ? end_step - start_step : 0;
    }
};

using IntervalMap = std::unordered_map<ResourceUUID, LiveInterval, ResourceUUIDHash>;

// Computes live intervals for all node outputs in a topological sequence.
//
// Algorithm (backward scan):
//   1. Walk topological_order, assign each node a step index.
//   2. For each connection, the source output is live from its definition step
//      to the maximum step of any consumer.
//   3. External outputs (chain tail, final output) are pinned to the end of
//      the sequence so they are never reclaimed.
//
// Complexity: O(N + E) where N = nodes, E = connections.
class LivenessAnalysis {
public:
    static IntervalMap compute_intervals(
        const std::vector<NodeId>& topological_order,
        const std::vector<Connection>& connections,
        const std::vector<ResourceUUID>& external_outputs);
};

} // namespace te::register_allocation
