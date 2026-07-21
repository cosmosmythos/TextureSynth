#pragma once

#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/ShaderVariantKey.hpp"
#include "engine/graphfusion/FusionGroup.hpp"
#include "engine/graphfusion/FusionGroupEmitter.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace te::fusion {

struct CompiledGroup {
    std::string glsl;
    std::string error;

    std::vector<NodeId> nodes;

    uint32_t param_base_slot = 0;
    uint32_t param_floats    = 0;

    struct ExternalInput {
        uint32_t slot;
        NodeId   src_node;
        uint32_t src_socket;
        NodeId   dst_node;
        uint32_t dst_socket;
    };
    std::vector<ExternalInput> external_inputs;

    NodeId   output_node   = 0;
    uint32_t output_socket = 0;

    uint32_t pass_index         = 0;
    uint32_t pass_count         = 1;
    uint32_t intermediate_count = 0;

    FusedVariantKey fused_key;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

struct CompiledGroupBundle {
    std::vector<CompiledGroup> groups;
    std::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

// Build a FusedVariantKey for the given group, capturing every field
// that affects the emitted GLSL. Equal keys produce identical SPIR-V.
// The key is stored on CompiledGroup and used by populate_groups_()
// to avoid recompiling shaders that are already cached on disk.
inline FusedVariantKey build_fused_key(const FusionGroup& group,
                                       const FusionContext& ctx,
                                       const GraphIR& ir,
                                       const NodeLibrary& lib) {
    FusedVariantKey key;
    std::unordered_map<NodeId, uint32_t> local_index;
    uint32_t total_sockets = 0;

    for (uint32_t li = 0; li < group.nodes.size(); ++li) {
        NodeId nid = group.nodes[li];
        local_index[nid] = li;
        const NodeType* t = ctx.node_type.at(nid);
        key.node_type_ids.push_back(t->id);
        key.input_counts.push_back(static_cast<uint32_t>(t->inputs.size()));
        key.param_socket_masks.push_back(0);
        if (ctx.bypassed_nodes.count(nid))
            key.bypass_mask |= (1u << li);
        total_sockets += static_cast<uint32_t>(t->inputs.size());
    }

    // feature_flags: low 3 bits = ChannelFormat, bits 3-4 = BitDepth
    NodeId output_node = group.nodes.empty() ? 0 : group.nodes.back();
    if (const auto* vn = ir.find(output_node)) {
        StorageFormat sf = resolve_node_storage(*vn, lib);
        uint32_t fmt   = static_cast<uint32_t>(sf.channels) & 0x7u;
        uint32_t depth = static_cast<uint32_t>(sf.depth) & 0x3u;
        key.feature_flags = fmt | (depth << 3);
    }

    // external_socket_masks: per-node bitmask of cross-group input sockets
    key.external_socket_masks.assign(group.nodes.size(), 0);
    for (const auto& ext : group.external_inputs) {
        auto li = local_index.find(ext.dst_node);
        if (li != local_index.end())
            key.external_socket_masks[li->second] |= (1u << ext.dst_socket);
    }

    // internal_producer_indices: for each socket, the local index of the
    // in-group node that feeds it, or UINT32_MAX for external/unconnected.
    key.internal_producer_indices.assign(total_sockets, UINT32_MAX);
    uint32_t flat = 0;
    for (uint32_t li = 0; li < group.nodes.size(); ++li) {
        NodeId nid = group.nodes[li];
        const NodeType* t = ctx.node_type.at(nid);
        auto dst_it = ctx.conns_by_dst.find(nid);
        for (uint32_t s = 0; s < t->inputs.size(); ++s) {
            NodeId src = 0;
            if (dst_it != ctx.conns_by_dst.end()) {
                for (const auto& conn : dst_it->second) {
                    const auto& [src_node, src_socket, dst_socket] = conn;
                    if (dst_socket == s) { src = src_node; break; }
                }
            }
            auto sli = local_index.find(src);
            if (sli != local_index.end())
                key.internal_producer_indices[flat] = sli->second;
            ++flat;
        }
    }

    return key;
}

inline CompiledGroupBundle compile_groups(
    const FusionGroupBundle& bundle,
    const GraphIR& ir,
    const FusionContext& ctx,
    const NodeLibrary& lib,
    uint32_t global_param_base = 0)
{
    CompiledGroupBundle result;
    result.groups.reserve(bundle.groups.size());

    for (size_t gi = 0; gi < bundle.groups.size(); ++gi) {
        const auto& group = bundle.groups[gi];
        CompiledGroup cg;
        cg.nodes = group.nodes;

        // emit GLSL
        auto emitted = emit_group(group, ir, ctx, static_cast<uint32_t>(gi), lib);
        if (!emitted.ok()) {
            cg.error = "group " + std::to_string(gi) + ": " + emitted.error;
            result.error = cg.error;
            result.groups.push_back(std::move(cg));
            return result;
        }
        cg.glsl = std::move(emitted.source);

        // param layout from context (already computed in build_context)
        uint32_t min_slot = UINT32_MAX;
        uint32_t total = 0;
        for (NodeId nid : group.nodes) {
            auto it = ctx.param_base.find(nid);
            if (it != ctx.param_base.end())
                min_slot = std::min(min_slot, it->second);
            auto fi = ctx.float_inputs.find(nid);
            auto pc = ctx.param_count.find(nid);
            if (fi != ctx.float_inputs.end() && pc != ctx.param_count.end())
                total += fi->second + pc->second;
        }
        cg.param_base_slot = (min_slot == UINT32_MAX) ? 0 : min_slot;
        cg.param_floats = total;

        // external inputs
        cg.external_inputs.reserve(group.external_inputs.size());
        for (const auto& ext : group.external_inputs)
            cg.external_inputs.push_back({ext.slot, ext.src_node, ext.src_socket,
                                          ext.dst_node, ext.dst_socket});

        // output: tail node
        if (!group.nodes.empty()) {
            cg.output_node = group.nodes.back();
            cg.output_socket = 0;
        }

        cg.pass_index         = group.pass_index;
        cg.pass_count         = group.pass_count;
        cg.intermediate_count = group.intermediate_count;

        cg.fused_key = build_fused_key(group, ctx, ir, lib);

        result.groups.push_back(std::move(cg));
    }

    return result;
}

} // namespace te::fusion
