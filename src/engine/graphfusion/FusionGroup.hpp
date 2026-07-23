#pragma once

#include "engine/GraphIR.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/Graph.hpp"
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace te::fusion {

struct NodePassInfo {
    uint32_t pass_index = 0;
    uint32_t pass_count = 1;
};

struct FusionGroup {
    std::vector<NodeId> nodes;
    std::unordered_map<NodeId, NodePassInfo> node_pass_map;
    uint32_t pass_index = 0;
    uint32_t pass_count = 1;
    uint32_t intermediate_count = 0;
    struct ExternalInput {
        NodeId src_node;
        uint32_t src_socket;
        NodeId dst_node;
        uint32_t dst_socket;
        uint32_t slot;
    };
    std::vector<ExternalInput> external_inputs;
};
struct FusionGroupBundle { std::vector<FusionGroup> groups; };

struct ExpandedNode {
    NodeId node_id;
    uint32_t pass_index;
    uint32_t pass_count;
};

struct FusionContext {
    std::unordered_map<NodeId, const NodeType*> node_type;

    // dst_node → [(src_node, src_socket, dst_socket)]
    std::unordered_map<NodeId, std::vector<std::tuple<NodeId, uint32_t, uint32_t>>> conns_by_dst;
    // src_node → [(dst_node, dst_socket, src_socket)]
    std::unordered_map<NodeId, std::vector<std::tuple<NodeId, uint32_t, uint32_t>>> conns_by_src;

    std::unordered_map<NodeId, uint32_t> param_base;
    std::unordered_map<NodeId, uint32_t> float_inputs;
    std::unordered_map<NodeId, uint32_t> param_count;
    uint32_t total_param_floats = 0;

    std::unordered_set<NodeId> bypassed_nodes;
};

inline FusionContext build_context(const GraphIR& ir, const NodeLibrary& lib) {
    FusionContext ctx;

    ctx.node_type.reserve(ir.nodes.size());
    for (const auto& validated_node : ir.nodes) {
        ctx.node_type[validated_node.id] = lib.find(validated_node.type_id);
        if (validated_node.bypassed) {
            ctx.bypassed_nodes.insert(validated_node.id);
        }
    }

    ctx.conns_by_dst.reserve(ir.connections.size());
    ctx.conns_by_src.reserve(ir.connections.size());
    for (const auto& c : ir.connections) {
        ctx.conns_by_dst[c.dst_node].push_back({c.src_node, c.src_socket, c.dst_socket});
        ctx.conns_by_src[c.src_node].push_back({c.dst_node, c.dst_socket, c.src_socket});
    }

    uint32_t next_slot = 0;
    for (NodeId id : ir.eval_order) {
        auto type_it = ctx.node_type.find(id);
        if (type_it == ctx.node_type.end() || !type_it->second) continue;
        const auto* type = type_it->second;

        uint32_t n_float = 0;
        for (const auto& inp : type->inputs)
            if (inp.type == SocketType::Float) ++n_float;

        ctx.param_base[id]   = next_slot;
        ctx.float_inputs[id] = n_float;
        ctx.param_count[id]  = static_cast<uint32_t>(type->params.size());
        next_slot += n_float + ctx.param_count[id];
    }
    ctx.total_param_floats = next_slot;

    return ctx;
}

[[nodiscard]] inline std::vector<ExpandedNode> expand_multipass(const std::vector<NodeId>& eval_order, const FusionContext& ctx) {
    std::vector<ExpandedNode> expanded;
    expanded.reserve(eval_order.size());
    for (NodeId node_id : eval_order) {
        auto type_it = ctx.node_type.find(node_id);
        if (type_it == ctx.node_type.end()) continue;
        const auto* type = type_it->second;
        for (uint32_t pass = 0; pass < type->pass_count; ++pass) {
            expanded.push_back({node_id, pass, type->pass_count});
        }
    }
    return expanded;
}

[[nodiscard]] inline std::optional<SocketType> get_connection_type(NodeId source, NodeId dest, const FusionContext& ctx) {
    auto dest_conns = ctx.conns_by_dst.find(dest);
    if (dest_conns == ctx.conns_by_dst.end()) return std::nullopt;

    for (const auto& [src_node, src_socket, dst_socket] : dest_conns->second) {
        if (src_node != source) continue;
        auto type_it = ctx.node_type.find(dest);
        if (type_it == ctx.node_type.end()) return std::nullopt;
        return type_it->second->inputs[dst_socket].type;
    }
    return std::nullopt;
}

inline bool is_connected(NodeId source, NodeId dest, const FusionContext& ctx) {
    auto type = get_connection_type(source, dest, ctx);
    return type.has_value() && *type != SocketType::Sampler2D;
}

[[nodiscard]] inline FusionGroupBundle group_nodes(const GraphIR& ir, const FusionContext& ctx) {
    FusionGroupBundle fused;
    auto expanded = expand_multipass(ir.eval_order, ctx);
    if (expanded.empty()) return fused;

    FusionGroup current_group;
    current_group.nodes.push_back(expanded[0].node_id);
    current_group.node_pass_map[expanded[0].node_id] = {expanded[0].pass_index, expanded[0].pass_count};
    current_group.pass_index = expanded[0].pass_index;
    current_group.pass_count = expanded[0].pass_count;
    {
        auto it = ctx.node_type.find(expanded[0].node_id);
        if (it != ctx.node_type.end() && it->second)
            current_group.intermediate_count = it->second->intermediate_count;
    }

    for (size_t i = 0; i + 1 < expanded.size(); ++i) {
        NodeId previous = expanded[i].node_id;
        NodeId current = expanded[i + 1].node_id;
        uint32_t current_pass = expanded[i + 1].pass_index;
        uint32_t current_count = expanded[i + 1].pass_count;

        bool conn = is_connected(previous, current, ctx);

        if (conn) {
            current_group.nodes.push_back(current);
            current_group.node_pass_map[current] = {current_pass, current_count};
        } else {
            fused.groups.push_back(std::move(current_group));
            current_group = FusionGroup();
            current_group.nodes.push_back(current);
            current_group.node_pass_map[current] = {current_pass, current_count};
            current_group.pass_index = expanded[i + 1].pass_index;
            current_group.pass_count = expanded[i + 1].pass_count;
            auto it = ctx.node_type.find(current);
            if (it != ctx.node_type.end() && it->second)
                current_group.intermediate_count = it->second->intermediate_count;
        }
    }

    fused.groups.push_back(std::move(current_group));

    return fused;
}

inline bool group_contains(const FusionGroup& group, NodeId id) {
    for (NodeId n : group.nodes)
        if (n == id) return true;
    return false;
}

inline void split_at_sampler2d_sources(FusionGroupBundle& fused, const FusionContext& ctx) {
    for (size_t group_idx = 0; group_idx < fused.groups.size(); ++group_idx) {
        auto& group = fused.groups[group_idx];
        if (group.nodes.size() <= 1) continue;

        for (size_t node_idx = 0; node_idx + 1 < group.nodes.size(); ++node_idx) {
            NodeId node = group.nodes[node_idx];
            auto source_it = ctx.conns_by_src.find(node);
            if (source_it == ctx.conns_by_src.end()) continue;

            bool needs_split = false;
            for (const auto& [dest_node, dest_socket, src_socket] : source_it->second) {
                if (group_contains(group, dest_node)) continue;
                auto type_it = ctx.node_type.find(dest_node);
                if (type_it == ctx.node_type.end()) continue;
                if (type_it->second->inputs[dest_socket].type == SocketType::Sampler2D) {
                    needs_split = true;
                    break;
                }
            }
            if (!needs_split) continue;

            FusionGroup tail_group;
            tail_group.nodes.assign(group.nodes.begin() + node_idx + 1, group.nodes.end());
            group.nodes.resize(node_idx + 1);
            fused.groups.insert(fused.groups.begin() + group_idx + 1, std::move(tail_group));
            --group_idx;
            break;
        }
    }
}

inline void merge_groups(FusionGroupBundle& fused, const FusionContext& ctx) {
    if (fused.groups.empty()) return;
    for (size_t index = fused.groups.size(); index > 0; --index) {
        size_t group_index = index - 1;
        if (fused.groups[group_index].nodes.empty()) continue;

        NodeId tail_node = fused.groups[group_index].nodes.back();

        size_t target_group = fused.groups.size();
        for (size_t dest_index = 0; dest_index < fused.groups.size(); ++dest_index) {
            if (dest_index == group_index || fused.groups[dest_index].nodes.empty()) continue;
            if (group_contains(fused.groups[dest_index], tail_node)) continue;

            for (NodeId node_in_dest : fused.groups[dest_index].nodes) {
                if (get_connection_type(tail_node, node_in_dest, ctx).has_value()) {
                    target_group = dest_index;
                    break;
                }
            }
            if (target_group != fused.groups.size()) break;
        }
        if (target_group == fused.groups.size()) continue;

        bool sampler2d = false;
        for (NodeId src_node : fused.groups[group_index].nodes) {
            auto source_it = ctx.conns_by_src.find(src_node);
            if (source_it == ctx.conns_by_src.end()) continue;
            for (const auto& [dest_node, dest_socket, src_socket] : source_it->second) {
                if (group_contains(fused.groups[group_index], dest_node)) continue;
                auto type_it = ctx.node_type.find(dest_node);
                if (type_it == ctx.node_type.end()) continue;
                if (type_it->second->inputs[dest_socket].type == SocketType::Sampler2D) {
                    sampler2d = true;
                    break;
                }
            }
            if (sampler2d) break;
        }
        if (sampler2d) continue;

        fused.groups[target_group].nodes.insert(
            fused.groups[target_group].nodes.begin(),
            fused.groups[group_index].nodes.begin(), fused.groups[group_index].nodes.end());
        for (const auto& [node_id, pass_info] : fused.groups[group_index].node_pass_map)
            fused.groups[target_group].node_pass_map[node_id] = pass_info;

        size_t erase_at = (target_group <= group_index) ? group_index + 1 : group_index;
        fused.groups.erase(fused.groups.begin() + erase_at);
        // Reverse loop: ++index offsets the decrement when erased group shifts elements left.
        if (erase_at <= group_index) ++index;
    }
}

inline void compute_external_inputs(FusionGroupBundle& fused, const FusionContext& ctx) {
    for (auto& group : fused.groups) {
        group.external_inputs.clear();
        uint32_t slot = 0;

        for (NodeId node_in_group : group.nodes) {
            auto type_it = ctx.node_type.find(node_in_group);
            if (type_it == ctx.node_type.end()) continue;
            const auto* type = type_it->second;

            auto connections = ctx.conns_by_dst.find(node_in_group);
            bool has_connections = (connections != ctx.conns_by_dst.end());

            // Cross-group connections: source lives outside this group.
            if (has_connections) {
                for (const auto& [src_node, src_socket, dst_socket] : connections->second) {
                    if (group_contains(group, src_node)) continue;
                    group.external_inputs.push_back({src_node, src_socket, node_in_group, dst_socket, slot++});
                }
            }

            // Unconnected sampler2D inputs: image bound at runtime via image_registry_.
            for (uint32_t s = 0; s < type->inputs.size(); ++s) {
                if (type->inputs[s].type != SocketType::Sampler2D) continue;

                bool already_connected = false;
                if (has_connections) {
                    for (const auto& [src_node, src_socket, dst_socket] : connections->second) {
                        if (dst_socket == s) { already_connected = true; break; }
                    }
                }
                if (!already_connected) {
                    group.external_inputs.push_back({0, 0, node_in_group, s, slot++});
                }
            }
        }
    }
}

} // namespace te::fusion
