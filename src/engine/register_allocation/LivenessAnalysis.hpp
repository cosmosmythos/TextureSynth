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
    // Template version: accepts Connection, ValidatedConnection, or any type
    // with src_node/src_socket/dst_node/dst_socket fields.
    template <typename ConnVec>
    static IntervalMap compute_intervals(
        const std::vector<NodeId>& topological_order,
        const ConnVec& connections,
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
};

} // namespace te::register_allocation
