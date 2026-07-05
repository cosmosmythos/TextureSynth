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

        result.groups.push_back(std::move(cg));
    }

    return result;
}

} // namespace te::fusion
