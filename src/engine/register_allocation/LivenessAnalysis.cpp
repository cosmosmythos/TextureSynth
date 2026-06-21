#include "engine/register_allocation/LivenessAnalysis.hpp"
#include <algorithm>

namespace te::register_allocation {

std::unordered_map<ResourceUUID, LiveInterval, ResourceUUIDHash> LivenessAnalysis::compute_intervals(
    const std::vector<NodeId>& topological_order,
    const std::vector<Connection>& connections,
    const std::vector<ResourceUUID>& external_outputs)
{
    std::unordered_map<NodeId, uint32_t> node_to_step;
    node_to_step.reserve(topological_order.size());
    for (uint32_t step = 0; step < static_cast<uint32_t>(topological_order.size()); ++step) {
        node_to_step[topological_order[step]] = step;
    }

    std::unordered_map<ResourceUUID, LiveInterval, ResourceUUIDHash> intervals;

    // Initialize intervals for all intermediate outputs.
    for (const auto& conn : connections) {
        const auto src_it = node_to_step.find(conn.src_node);
        if (src_it == node_to_step.end()) {
            continue;
        }

        ResourceUUID res{conn.src_node, conn.src_socket};
        if (intervals.find(res) == intervals.end()) {
            LiveInterval interval;
            interval.resource = res;
            interval.start_step = src_it->second;
            interval.end_step = src_it->second;
            intervals[res] = interval;
        }
    }

    // Extend intervals based on reads.
    for (const auto& conn : connections) {
        const auto src_it = node_to_step.find(conn.src_node);
        const auto dst_it = node_to_step.find(conn.dst_node);
        if (src_it == node_to_step.end() || dst_it == node_to_step.end()) {
            continue;
        }

        ResourceUUID res{conn.src_node, conn.src_socket};
        intervals[res].end_step = std::max(intervals[res].end_step, dst_it->second);
    }

    // Force external outputs to remain live until the end.
    const uint32_t end_bound = static_cast<uint32_t>(topological_order.size());
    for (const auto& res : external_outputs) {
        const auto src_it = node_to_step.find(res.node_id);
        if (src_it == node_to_step.end()) {
            continue;
        }

        auto it = intervals.find(res);
        if (it == intervals.end()) {
            LiveInterval interval;
            interval.resource = res;
            interval.start_step = src_it->second;
            interval.end_step = end_bound;
            intervals[res] = interval;
        } else {
            it->second.end_step = end_bound;
        }
    }

    return intervals;
}

} // namespace te::register_allocation
