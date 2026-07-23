#pragma once

#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/FusionGroup.hpp"
#include "engine/graphfusion/FusionGroupEmitter.hpp"
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

    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

struct CompiledGroupBundle {
    std::vector<CompiledGroup> groups;
    std::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

inline CompiledGroupBundle compile_groups(
    const FusionGroupBundle& bundle,
    const GraphIR& ir,
    const FusionContext& ctx,
    const NodeLibrary& lib,
    uint32_t global_param_base = 0)
{
    CompiledGroupBundle result;
    result.groups.reserve(bundle.groups.size());

    for (size_t group_idx = 0; group_idx < bundle.groups.size(); ++group_idx) {
        const auto& group = bundle.groups[group_idx];
        CompiledGroup compiled_group;
        compiled_group.nodes = group.nodes;

        // emit GLSL
        auto emitted = emit_group(group, ir, ctx, static_cast<uint32_t>(group_idx), lib);
        if (!emitted.ok()) {
            compiled_group.error = "group " + std::to_string(group_idx) + ": " + emitted.error;
            result.error = compiled_group.error;
            result.groups.push_back(std::move(compiled_group));
            return result;
        }
        compiled_group.glsl = std::move(emitted.source);

        // param layout from context (already computed in build_context)
        uint32_t min_slot = UINT32_MAX;
        uint32_t total = 0;
        for (NodeId node_id : group.nodes) {
            auto it = ctx.param_base.find(node_id);
            if (it != ctx.param_base.end())
                min_slot = std::min(min_slot, it->second);
            auto fi = ctx.float_inputs.find(node_id);
            auto pc = ctx.param_count.find(node_id);
            if (fi != ctx.float_inputs.end() && pc != ctx.param_count.end())
                total += fi->second + pc->second;
        }
        compiled_group.param_base_slot = (min_slot == UINT32_MAX) ? 0 : min_slot;
        compiled_group.param_floats = total;

        // external inputs
        compiled_group.external_inputs.reserve(group.external_inputs.size());
        for (const auto& ext : group.external_inputs)
            compiled_group.external_inputs.push_back({ext.slot, ext.src_node, ext.src_socket,
                                          ext.dst_node, ext.dst_socket});

        // output: tail node
        if (!group.nodes.empty()) {
            compiled_group.output_node = group.nodes.back();
            compiled_group.output_socket = 0;
        }

        compiled_group.pass_index         = group.pass_index;
        compiled_group.pass_count         = group.pass_count;
        compiled_group.intermediate_count = group.intermediate_count;

        result.groups.push_back(std::move(compiled_group));
    }

    return result;
}

} // namespace te::fusion
