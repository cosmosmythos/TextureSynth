#pragma once

#include "engine/Graph.hpp"
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace te::register_allocation {

// Undirected interference graph where nodes = ResourceUUIDs (node outputs)
// and edges connect any two resources whose live intervals overlap.
//
// Two resources that interfere cannot share the same GLSL local variable
// (or physical register on the GPU). The graph is built from LiveIntervals
// and consumed by GraphColorer.
class InterferenceGraph {
public:
    using NodeSet = std::unordered_set<ResourceUUID, ResourceUUIDHash>;

    void add_node(ResourceUUID res);
    void add_edge(ResourceUUID a, ResourceUUID b);

    [[nodiscard]] bool has_node(ResourceUUID res) const;
    [[nodiscard]] bool has_edge(ResourceUUID a, ResourceUUID b) const;
    [[nodiscard]] uint32_t degree(ResourceUUID res) const;

    [[nodiscard]] const NodeSet& neighbors(ResourceUUID res) const;
    [[nodiscard]] const std::vector<ResourceUUID>& nodes() const { return node_list_; }
    [[nodiscard]] uint32_t node_count() const { return static_cast<uint32_t>(node_list_.size()); }

    void remove_node(ResourceUUID res);

    // Build an interference graph from pre-computed live intervals.
    // Two resources interfere if their intervals overlap.
    // Complexity: O(N^2) where N = number of intervals. For the small
    // chain sizes in TextureSynth (typically 3-20 nodes) this is fine.
    static InterferenceGraph build_from_intervals(
        const std::unordered_map<ResourceUUID, struct LiveInterval, ResourceUUIDHash>& intervals);

private:
    std::vector<ResourceUUID> node_list_;
    std::unordered_set<ResourceUUID, ResourceUUIDHash> node_set_;
    std::unordered_map<ResourceUUID, NodeSet, ResourceUUIDHash> adjacency_;

    static const NodeSet empty_set_;
};

} // namespace te::register_allocation
