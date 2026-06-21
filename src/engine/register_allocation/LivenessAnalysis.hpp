#pragma once

#include "engine/Graph.hpp"
#include <unordered_map>
#include <vector>

namespace te::register_allocation {

struct LiveInterval {
    ResourceUUID resource;
    uint32_t start_step = 0;
    uint32_t end_step = 0;

    [[nodiscard]] bool overlaps(const LiveInterval& other) const noexcept {
        return start_step <= other.end_step && other.start_step <= end_step;
    }
};

class LivenessAnalysis {
public:
    static std::unordered_map<ResourceUUID, LiveInterval, ResourceUUIDHash> compute_intervals(
        const std::vector<NodeId>& topological_order,
        const std::vector<Connection>& connections,
        const std::vector<ResourceUUID>& external_outputs);
};

} // namespace te::register_allocation
