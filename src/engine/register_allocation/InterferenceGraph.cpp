#include "engine/register_allocation/InterferenceGraph.hpp"
#include "engine/register_allocation/LivenessAnalysis.hpp"
#include <algorithm>

namespace te::register_allocation {

const InterferenceGraph::NodeSet InterferenceGraph::empty_set_;

void InterferenceGraph::add_node(ResourceUUID res) {
    if (node_set_.insert(res).second) {
        node_list_.push_back(res);
        adjacency_.try_emplace(res);
    }
}

void InterferenceGraph::add_edge(ResourceUUID a, ResourceUUID b) {
    if (a == b) return;
    add_node(a);
    add_node(b);
    adjacency_[a].insert(b);
    adjacency_[b].insert(a);
}

bool InterferenceGraph::has_node(ResourceUUID res) const {
    return node_set_.find(res) != node_set_.end();
}

bool InterferenceGraph::has_edge(ResourceUUID a, ResourceUUID b) const {
    auto it = adjacency_.find(a);
    if (it == adjacency_.end()) return false;
    return it->second.find(b) != it->second.end();
}

uint32_t InterferenceGraph::degree(ResourceUUID res) const {
    auto it = adjacency_.find(res);
    return it != adjacency_.end() ? static_cast<uint32_t>(it->second.size()) : 0;
}

const InterferenceGraph::NodeSet& InterferenceGraph::neighbors(ResourceUUID res) const {
    auto it = adjacency_.find(res);
    return it != adjacency_.end() ? it->second : empty_set_;
}

void InterferenceGraph::remove_node(ResourceUUID res) {
    auto adj_it = adjacency_.find(res);
    if (adj_it == adjacency_.end()) return;

    // Remove edges from neighbors pointing back to this node.
    for (const auto& neighbor : adj_it->second) {
        auto n_it = adjacency_.find(neighbor);
        if (n_it != adjacency_.end())
            n_it->second.erase(res);
    }

    adjacency_.erase(adj_it);
    node_set_.erase(res);
    node_list_.erase(
        std::remove(node_list_.begin(), node_list_.end(), res),
        node_list_.end());
}

InterferenceGraph InterferenceGraph::build_from_intervals(
    const std::unordered_map<ResourceUUID, LiveInterval, ResourceUUIDHash>& intervals)
{
    InterferenceGraph graph;

    // Collect all intervals into a sortable vector.
    std::vector<const LiveInterval*> sorted;
    sorted.reserve(intervals.size());
    for (const auto& [res, iv] : intervals) {
        graph.add_node(res);
        sorted.push_back(&iv);
    }

    // Sort by start_step for sweep-line efficiency.
    std::sort(sorted.begin(), sorted.end(),
        [](const LiveInterval* a, const LiveInterval* b) {
            return a->start_step < b->start_step;
        });

    // Sweep: for each pair, add edge if intervals overlap.
    // O(N^2) but N is small (chain size). A sweep-line with an active set
    // can reduce this to O(N log N + E), but the chains are typically < 50 nodes.
    for (size_t i = 0; i < sorted.size(); ++i) {
        for (size_t j = i + 1; j < sorted.size(); ++j) {
            // Since sorted by start, if j.start > i.end, no further j can overlap i.
            if (sorted[j]->start_step > sorted[i]->end_step)
                break;
            if (sorted[i]->overlaps(*sorted[j]))
                graph.add_edge(sorted[i]->resource, sorted[j]->resource);
        }
    }

    return graph;
}

} // namespace te::register_allocation
