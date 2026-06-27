#include "engine/register_allocation/GraphColorer.hpp"
#include <algorithm>
#include <limits>
#include <stack>
#include <unordered_set>

namespace te::register_allocation {

// ---------------------------------------------------------------------------
// Chaitin-Briggs: Optimistic Graph Coloring
// ---------------------------------------------------------------------------
//
// Classic compiler algorithm adapted for our use case:
//
// Phase 1 — Simplify:
//   While the graph is non-empty:
//     a) If any node has degree < K, remove it and push onto stack.
//        (Guaranteed colorable: when we pop it, at most K-1 neighbors
//         are already colored, so at least one color is free.)
//     b) Else (all nodes have degree >= K): pick the node with the
//        highest degree and push it anyway (optimistic — Briggs' insight:
//        when we pop it during Select, its neighbors may have received
//        fewer than K distinct colors, leaving room).
//
// Phase 2 — Select:
//   Pop nodes from the stack and assign the lowest color not used by
//   any already-colored neighbor. If no color < K is available, the
//   node is spilled.
//
// The optimistic approach avoids eagerly spilling nodes that turn out
// to be colorable. This is critical for GPU shaders where spilling
// (writing a vec4 to shared memory) is extremely expensive.

ColoringResult GraphColorer::color_chaitin_briggs(
    const InterferenceGraph& graph,
    uint32_t budget)
{
    ColoringResult result;
    if (graph.node_count() == 0) return result;

    // Work on a mutable copy.
    InterferenceGraph work = graph;
    std::stack<ResourceUUID> stack;

    // Simplify phase.
    while (work.node_count() > 0) {
        bool found_low_degree = false;

        // Scan for any node with degree < budget.
        for (const auto& node : work.nodes()) {
            if (work.degree(node) < budget) {
                stack.push(node);
                work.remove_node(node);
                found_low_degree = true;
                break;
            }
        }

        if (!found_low_degree) {
            // Optimistic spill: pick node with highest degree.
            ResourceUUID victim{};
            uint32_t max_deg = 0;
            for (const auto& node : work.nodes()) {
                uint32_t d = work.degree(node);
                if (d >= max_deg) {
                    max_deg = d;
                    victim = node;
                }
            }
            stack.push(victim);
            work.remove_node(victim);
        }
    }

    // Select phase: pop from stack, assign colors.
    while (!stack.empty()) {
        ResourceUUID node = stack.top();
        stack.pop();

        // Gather colors used by already-colored neighbors.
        std::unordered_set<uint32_t> used_colors;
        for (const auto& neighbor : graph.neighbors(node)) {
            auto it = result.assignment.find(neighbor);
            if (it != result.assignment.end())
                used_colors.insert(it->second);
        }

        // Find lowest available color.
        uint32_t color = 0;
        while (used_colors.find(color) != used_colors.end())
            ++color;

        if (color < budget) {
            result.assignment[node] = color;
            if (color + 1 > result.colors_used)
                result.colors_used = color + 1;
        } else {
            // Truly cannot color within budget — spill.
            result.spilled.push_back(node);
        }
    }

    return result;
}


// ---------------------------------------------------------------------------
// Linear Scan: Greedy Interval Coloring
// ---------------------------------------------------------------------------
//
// From Poletto & Sarkar (1999), adapted for our DAG:
//
// 1. Sort live intervals by start_step.
// 2. Maintain an "active" set of intervals currently occupying a register.
// 3. For each interval in sorted order:
//    a) Expire: remove all active intervals whose end_step < current start_step.
//       Return their registers to the free pool.
//    b) If free pool is non-empty: assign the lowest-numbered free register.
//    c) Else if budget is exceeded: spill the active interval with the
//       latest end_step (frees a register for the shorter-lived current
//       interval, reducing future pressure).
//
// For interval graphs (which topological-order liveness produces), this
// is optimal — the greedy coloring matches the chromatic number.

ColoringResult GraphColorer::color_linear_scan(
    const IntervalMap& intervals,
    const std::vector<NodeId>& topological_order,
    uint32_t budget)
{
    ColoringResult result;
    if (intervals.empty()) return result;

    // Build sorted vector of intervals by start_step, breaking ties by end_step.
    struct SortedInterval {
        ResourceUUID resource;
        uint32_t start;
        uint32_t end;
    };
    std::vector<SortedInterval> sorted;
    sorted.reserve(intervals.size());
    for (const auto& [res, iv] : intervals)
        sorted.push_back({res, iv.start_step, iv.end_step});

    std::sort(sorted.begin(), sorted.end(),
        [](const SortedInterval& a, const SortedInterval& b) {
            if (a.start != b.start) return a.start < b.start;
            return a.end < b.end;
        });

    // Active set: intervals currently occupying a register, sorted by end_step.
    struct ActiveEntry {
        ResourceUUID resource;
        uint32_t end;
        uint32_t reg;
    };
    std::vector<ActiveEntry> active;

    // Free register pool.
    std::vector<uint32_t> free_regs;

    uint32_t next_reg = 0;

    for (const auto& si : sorted) {
        // Expire old intervals.
        auto new_end = std::remove_if(active.begin(), active.end(),
            [&](const ActiveEntry& ae) {
                if (ae.end < si.start) {
                    free_regs.push_back(ae.reg);
                    return true;
                }
                return false;
            });
        active.erase(new_end, active.end());

        // Sort free_regs so we assign the lowest available.
        std::sort(free_regs.begin(), free_regs.end());

        if (!free_regs.empty()) {
            // Reuse the lowest free register.
            uint32_t reg = free_regs.front();
            free_regs.erase(free_regs.begin());
            result.assignment[si.resource] = reg;
            active.push_back({si.resource, si.end, reg});
        } else if (next_reg < budget) {
            // Allocate a new register.
            uint32_t reg = next_reg++;
            result.assignment[si.resource] = reg;
            active.push_back({si.resource, si.end, reg});
        } else {
            // Budget exceeded. Spill the active interval with the latest end.
            auto victim_it = std::max_element(active.begin(), active.end(),
                [](const ActiveEntry& a, const ActiveEntry& b) {
                    return a.end < b.end;
                });

            if (victim_it != active.end() && victim_it->end > si.end) {
                // Spill the longer-lived victim, give its register to current.
                result.spilled.push_back(victim_it->resource);
                result.assignment.erase(victim_it->resource);
                uint32_t freed_reg = victim_it->reg;
                active.erase(victim_it);

                result.assignment[si.resource] = freed_reg;
                active.push_back({si.resource, si.end, freed_reg});
            } else {
                // Current interval is the longest — spill it directly.
                result.spilled.push_back(si.resource);
            }
        }
    }

    result.colors_used = next_reg;

    // Assign shared memory slots to spilled resources.
    uint32_t next_shared_slot = 0;
    for (auto& rid : result.spilled)
        result.spilled_assignment[rid] = next_shared_slot++;
    result.shared_slot_count = next_shared_slot;

    return result;
}

} // namespace te::register_allocation
