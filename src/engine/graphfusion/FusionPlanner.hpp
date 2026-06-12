#pragma once

#include "DAG.hpp"
#include "RegisterAllocator.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
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
        FusionPlan<NodeId> result;

        if (active_path.empty()) {
            return result;
        }

        if (!is_valid_path(dag, active_path)) {
            result.valid = false;
            return result;
        }

        reg::RegisterAllocator allocator(budget_);
        for (const auto& node_id : active_path) {
            (void)node_id;
            if (!allocator.add_node("generic")) {
                break;
            }
        }

        if (allocator.used() <= budget_) {
            result.groups.push_back(make_single_group(active_path, allocator.used()));
            result.total_estimated_registers = allocator.used();
            return result;
        }

        result.needs_split = true;
        result.groups = split_path(active_path);

        for (const auto& group : result.groups) {
            result.total_estimated_registers += group.estimated_registers;
        }

        return result;
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
        result.groups = split_path(active_path);

        for (const auto& group : result.groups) {
            result.total_estimated_registers += group.estimated_registers;
        }

        return result;
    }

private:
    template <typename NodeId>
    [[nodiscard]] static bool is_valid_path(
        const dag::DAG<NodeId>& dag,
        const std::vector<NodeId>& path)
    {
        for (std::size_t i = 0; i < path.size(); ++i) {
            if (!dag.has_node(path[i])) {
                return false;
            }
            if (i > 0 && !dag.has_edge(path[i - 1], path[i])) {
                return false;
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
        const std::vector<NodeId>& path) const
    {
        std::vector<FusionGroup<NodeId>> groups;
        groups.reserve(path.size());

        reg::RegisterAllocator allocator(budget_);
        FusionGroup<NodeId> current;
        current.id = 0;

        for (const auto& node_id : path) {
            if (!current.nodes.empty() && allocator.would_exceed("generic")) {
                current.output_node = current.nodes.back();
                current.split_point = current.nodes.back();
                current.estimated_registers = allocator.used();
                groups.push_back(std::move(current));

                current = FusionGroup<NodeId>{};
                current.id = static_cast<std::uint32_t>(groups.size());
                allocator.reset();
            }

            current.nodes.push_back(node_id);
            (void)allocator.add_node("generic");
        }

        if (!current.nodes.empty()) {
            current.output_node = current.nodes.back();
            current.estimated_registers = allocator.used();
            groups.push_back(std::move(current));
        }

        return groups;
    }

    std::uint32_t budget_;
};


} // namespace te::fusion
