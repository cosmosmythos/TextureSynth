#pragma once

#include "engine/Graph.hpp"
#include "engine/register_allocation/InterferenceGraph.hpp"
#include "engine/register_allocation/LivenessAnalysis.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace te::register_allocation {

// Result of coloring: maps each ResourceUUID to a color (register index).
struct ColoringResult {
    // Maps resource -> assigned color. Color 0 is valid.
    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> assignment;

    // Number of distinct colors used (= number of GLSL vec4 locals needed).
    uint32_t colors_used = 0;

    // Resources that could not be colored within the budget (need spill/split).
    std::vector<ResourceUUID> spilled;

    // Maps spilled resource -> shared memory slot index.
    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> spilled_assignment;

    // Total shared memory slots needed (= spilled.size()).
    uint32_t shared_slot_count = 0;

    [[nodiscard]] bool has_spills() const { return !spilled.empty(); }
};

// Graph coloring register allocator implementing two strategies:
//
// 1. Chaitin-Briggs (optimistic coloring):
//    - Simplify: remove nodes with degree < K, push onto stack.
//    - Optimistic spill: if all nodes have degree >= K, push highest-degree
//      node anyway (Briggs' optimistic improvement over Chaitin).
//    - Select: pop from stack, assign lowest available color.
//    - If a node truly cannot be colored during select, mark it spilled.
//
// 2. Linear scan (greedy interval coloring):
//    - Sort intervals by start point.
//    - Greedily assign the lowest-numbered register whose previous user
//      has already expired. Falls back to a new register if none available.
//    - Spills the interval with the latest end point if budget is exceeded.
//
// Both strategies are static methods — no instance state needed.
class GraphColorer {
public:
    // Chaitin-Briggs graph coloring on an interference graph.
    // K = register budget. Returns assignment with at most K colors
    // (or spills if the chromatic number exceeds K).
    static ColoringResult color_chaitin_briggs(
        const InterferenceGraph& graph,
        uint32_t budget);

    // Linear scan coloring on sorted intervals.
    // Faster than Chaitin-Briggs, optimal for interval graphs (which
    // our topological-order liveness produces — a perfect graph).
    static ColoringResult color_linear_scan(
        const IntervalMap& intervals,
        const std::vector<NodeId>& topological_order,
        uint32_t budget);
};

} // namespace te::register_allocation
