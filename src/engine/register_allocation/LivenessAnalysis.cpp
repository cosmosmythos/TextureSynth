#include "engine/register_allocation/LivenessAnalysis.hpp"

namespace te::register_allocation {

IntervalMap LivenessAnalysis::compute_intervals(
    const std::vector<NodeId>& topological_order,
    const std::vector<Connection>& connections,
    const std::vector<ResourceUUID>& external_outputs)
{
    std::unordered_map<NodeId, uint32_t> node_to_step;
    node_to_step.reserve(topological_order.size());
    for (uint32_t step = 0; step < static_cast<uint32_t>(topological_order.size()); ++step)
        node_to_step[topological_order[step]] = step;

    IntervalMap intervals;

    // Pass 1: create intervals at definition site.
    for (const auto& conn : connections) {
        auto src_it = node_to_step.find(conn.src_node);
        if (src_it == node_to_step.end()) continue;

        ResourceUUID res{conn.src_node, conn.src_socket};
        if (intervals.find(res) == intervals.end()) {
            intervals[res] = LiveInterval{res, src_it->second, src_it->second};
        }
    }

    // Pass 2: extend intervals at each read site.
    for (const auto& conn : connections) {
        auto src_it = node_to_step.find(conn.src_node);
        auto dst_it = node_to_step.find(conn.dst_node);
        if (src_it == node_to_step.end() || dst_it == node_to_step.end()) continue;

        ResourceUUID res{conn.src_node, conn.src_socket};
        auto& iv = intervals[res];
        if (dst_it->second > iv.end_step)
            iv.end_step = dst_it->second;
    }

    // Pass 3: pin external outputs to end-of-sequence.
    const uint32_t end_bound = static_cast<uint32_t>(topological_order.size());
    for (const auto& res : external_outputs) {
        auto src_it = node_to_step.find(res.node_id);
        if (src_it == node_to_step.end()) continue;

        auto it = intervals.find(res);
        if (it == intervals.end())
            intervals[res] = LiveInterval{res, src_it->second, end_bound};
        else
            it->second.end_step = end_bound;
    }

    return intervals;
}

} // namespace te::register_allocation
