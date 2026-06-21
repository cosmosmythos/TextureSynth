#pragma once

#include "DAG.hpp"
#include "RegisterAllocator.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace te::fusion {

template <typename NodeId>
struct FusionGroup {
    std::uint32_t id = 0;
    std::vector<NodeId> nodes;
    std::optional<NodeId> output_node;
    std::optional<NodeId> split_point;
    std::uint64_t estimated_registers = 0;
};

template <typename NodeId>
struct FusionPlan {
    std::vector<FusionGroup<NodeId>> groups;
    bool needs_split = false;
    std::uint64_t total_estimated_registers = 0;
    bool valid = true;
};


class FusionPlanner {
public:
    explicit FusionPlanner(std::uint32_t reg_budget = reg::RegisterAllocator::DEFAULT_BUDGET) noexcept
        : budget_(reg_budget) {}

    template <typename NodeId>
    [[nodiscard]] FusionPlan<NodeId> plan(
        const dag::DAG<NodeId>& dag,
        const std::vector<NodeId>& active_path) const
    {
        std::vector<std::uint32_t> default_costs(active_path.size(), 5);
        return plan(dag, active_path, default_costs);
    }

    // plan() with pre-computed per-node register costs (preferred).
    template <typename NodeId>
    [[nodiscard]] FusionPlan<NodeId> plan(
        const dag::DAG<NodeId>& dag,
        const std::vector<NodeId>& active_path,
        const std::vector<std::uint32_t>& per_node_costs) const
    {
        FusionPlan<NodeId> result;

        if (active_path.empty()) {
            return result;
        }

        if (!is_valid_path(dag, active_path)) {
            result.valid = false;
            return result;
        }

        reg::RegisterAllocator allocator(budget_);
        for (std::size_t i = 0; i < active_path.size(); ++i) {
            const std::uint32_t cost = (i < per_node_costs.size()) ? per_node_costs[i] : 5;
            if (!allocator.add_node(reg::NodeRegCost{cost, 0, 0, 0, 0})) {
                break;
            }
        }

        if (allocator.used() <= budget_) {
            result.groups.push_back(make_single_group(active_path, allocator.used()));
            result.total_estimated_registers = allocator.used();
            return result;
        }

        result.needs_split = true;
        result.groups = split_path(dag, active_path, per_node_costs);

        for (const auto& group : result.groups) {
            result.total_estimated_registers += group.estimated_registers;
        }

        return result;
    }

private:
    // Validate that every node in `path` exists in `dag` and that the path
    // respects a topological order (no node appears before one of its
    // ancestors in the DAG). The path is NOT required to be a linear chain --
    // active paths from ActivePathTracer are topological orderings of a
    // sub-DAG, which may include siblings and fan-out. Requiring direct
    // edges between consecutive path[i-1] -> path[i] was wrong: it rejected
    // valid fan-out graphs (e.g. two noise sources feeding a blend).
    template <typename NodeId>
    [[nodiscard]] static bool is_valid_path(
        const dag::DAG<NodeId>& dag,
        const std::vector<NodeId>& path)
    {
        std::unordered_set<NodeId> seen;
        seen.reserve(path.size());
        for (std::size_t i = 0; i < path.size(); ++i) {
            if (!dag.has_node(path[i])) return false;
            if (!seen.insert(path[i]).second) return false;  // duplicate
            // Any ancestor of path[i] must already have appeared earlier.
            for (NodeId pred : dag.predecessors(path[i])) {
                if (seen.count(pred) == 0) return false;
            }
        }
        return true;
    }

    template <typename NodeId>
    [[nodiscard]] static FusionGroup<NodeId> make_single_group(
        const std::vector<NodeId>& path,
        std::uint64_t estimated_registers)
    {
        FusionGroup<NodeId> group;
        group.id = 0;
        group.nodes = path;
        group.output_node = path.back();
        group.estimated_registers = estimated_registers;
        return group;
    }

    template <typename NodeId>
    [[nodiscard]] std::vector<FusionGroup<NodeId>> split_path(
        const dag::DAG<NodeId>& dag,
        const std::vector<NodeId>& path,
        const std::vector<std::uint32_t>& per_node_costs) const
    {
        std::vector<FusionGroup<NodeId>> groups;
        groups.reserve(path.size());

        reg::RegisterAllocator allocator(budget_);
        std::unordered_set<NodeId> active_path_set(path.begin(), path.end());

        size_t start_idx = 0;
        while (start_idx < path.size()) {
            size_t best_end_idx = start_idx;
            uint64_t best_group_cost = 0;

            allocator.reset();
            for (size_t end_idx = start_idx; end_idx < path.size(); ++end_idx) {
                uint32_t cost = (end_idx < per_node_costs.size()) ? per_node_costs[end_idx] : 5;

                if (end_idx > start_idx && allocator.would_exceed(reg::NodeRegCost{cost, 0, 0, 0, 0})) {
                    break;
                }

                (void)allocator.add_node(reg::NodeRegCost{cost, 0, 0, 0, 0});

                // Check group validity:
                // For all intermediate nodes (start_idx <= i < end_idx),
                // their successors in the active path must be inside the group.
                bool valid = true;
                std::unordered_set<NodeId> group_set(path.begin() + start_idx, path.begin() + end_idx + 1);

                for (size_t i = start_idx; i < end_idx; ++i) {
                    NodeId intermediate = path[i];
                    for (NodeId succ : dag.successors(intermediate)) {
                        if (active_path_set.count(succ) && !group_set.count(succ)) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) break;
                }

                if (valid) {
                    best_end_idx = end_idx;
                    best_group_cost = allocator.used();
                }
            }

            FusionGroup<NodeId> group;
            group.id = static_cast<std::uint32_t>(groups.size());
            group.nodes.assign(path.begin() + start_idx, path.begin() + best_end_idx + 1);
            group.output_node = path[best_end_idx];
            if (best_end_idx < path.size() - 1) {
                group.split_point = path[best_end_idx];
            }
            group.estimated_registers = best_group_cost;
            groups.push_back(std::move(group));

            start_idx = best_end_idx + 1;
        }

        return groups;
    }

    std::uint32_t budget_;
};


} // namespace te::fusion
