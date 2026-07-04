#pragma once

#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/FusionGroup.hpp"
#include "engine/graphfusion/GlslBuilder.hpp"
#include <string>
#include <unordered_set>
#include <vector>

namespace te::fusion {

struct GroupEmitResult {
    std::string source;
    std::string error;
    uint32_t external_inputs = 0;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

inline GroupEmitResult emit_group(
    const FusionGroup& group,
    const GraphIR& ir,
    const FusionContext& ctx,
    uint32_t group_index)
{
    GroupEmitResult result;
    if (group.nodes.empty()) {
        result.error = "empty group";
        return result;
    }

    // node_id → position in this group
    std::unordered_map<NodeId, size_t> node_index;
    for (size_t i = 0; i < group.nodes.size(); ++i)
        node_index[group.nodes[i]] = i;

    // external input: src_node → slot
    std::unordered_map<NodeId, uint32_t> ext_slot_for_src;
    for (const auto& ext : group.external_inputs)
        ext_slot_for_src[ext.src_node] = ext.slot;

    glsl::GlslBuilder builder;
    builder.add_header(glsl::compute_header());
    builder.add_function(glsl::format_helpers());

    std::unordered_set<std::string> emitted_funcs;
    uint32_t param_offset = 0;

    builder.main_begin();
    builder.statement("if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;");

    std::string tail_var = "_g" + std::to_string(group_index) + "_out";

    for (size_t i = 0; i < group.nodes.size(); ++i) {
        NodeId nid = group.nodes[i];
        auto it = ctx.node_type.find(nid);
        const auto* type = (it != ctx.node_type.end()) ? it->second : nullptr;
        if (!type) {
            result.error = "node " + std::to_string(nid) + " not found";
            return result;
        }

        if (emitted_funcs.insert(type->glsl_function).second)
            builder.add_function(type->glsl_function);

        bool is_multi_output = type->outputs.size() > 1;

        std::string out_var = (i == group.nodes.size() - 1)
            ? tail_var
            : "_n" + std::to_string(i);

        if (is_multi_output) {
            for (uint32_t o = 0; o < type->outputs.size(); ++o)
                builder.declare_local(out_var + "_out" + std::to_string(o));
        } else {
            builder.declare_local(out_var);
        }

        std::vector<std::string> args;
        args.push_back("uv");

        uint32_t float_input_idx = 0;
        auto dst_it = ctx.conns_by_dst.find(nid);
        for (uint32_t s = 0; s < type->inputs.size(); ++s) {
            NodeId src = 0;
            if (dst_it != ctx.conns_by_dst.end()) {
                for (const auto& [src_node, src_socket, dst_socket] : dst_it->second) {
                    if (dst_socket == s) { src = src_node; break; }
                }
            }

            bool is_sampler = (type->inputs[s].type == SocketType::Sampler2D);
            bool is_float   = (type->inputs[s].type == SocketType::Float);

            if (src != 0 && node_index.count(src)) {
                std::string src_var = "_n" + std::to_string(node_index.at(src));
                const auto* src_type = ctx.node_type.count(src) ? ctx.node_type.at(src) : nullptr;
                if (src_type && src_type->outputs.size() > 1) {
                    auto src_ext_it = ctx.conns_by_dst.find(nid);
                    uint32_t src_socket_idx = 0;
                    if (src_ext_it != ctx.conns_by_dst.end()) {
                        for (const auto& [sn, ss, ds] : src_ext_it->second) {
                            if (sn == src && ds == s) { src_socket_idx = ss; break; }
                        }
                    }
                    src_var += "_out" + std::to_string(src_socket_idx);
                } else if (is_float) {
                    src_var += ".r";
                }
                args.push_back(std::move(src_var));
            } else if (src != 0) {
                uint32_t slot = 0;
                auto sit = ext_slot_for_src.find(src);
                if (sit != ext_slot_for_src.end()) slot = sit->second;
                if (is_sampler) {
                    args.push_back("TSTexture(" + std::to_string(slot) +
                        ", 1.0 / vec2(textureSize(u_sampled[nonuniformEXT("
                        "pc.in_sampled_slots[" + std::to_string(slot) + "])], 0)))");
                } else {
                    std::string ext_var = "_ext_" + std::to_string(slot);
                    builder.declare_external(ext_var, slot);
                    if (is_float) ext_var += ".r";
                    args.push_back(std::move(ext_var));
                }
            } else {
                if (is_float) {
                    args.push_back("node_params[pc.param_ring_idx].v[pc.param_base_slot + " +
                                   std::to_string(param_offset + float_input_idx) + "]");
                    ++float_input_idx;
                } else {
                    const auto& d = type->inputs[s].default_vec4;
                    args.push_back("vec4(" + std::to_string(d[0]) + ", " +
                                   std::to_string(d[1]) + ", " + std::to_string(d[2]) + ", " +
                                   std::to_string(d[3]) + ")");
                }
            }
        }

        for (uint32_t p = 0; p < type->params.size(); ++p) {
            args.push_back("node_params[pc.param_ring_idx].v[pc.param_base_slot + " +
                           std::to_string(param_offset + ctx.float_inputs.at(nid) + p) + "]");
        }

        if (is_multi_output) {
            for (uint32_t o = 0; o < type->outputs.size(); ++o)
                args.push_back(out_var + "_out" + std::to_string(o));
            builder.call_void("node_" + type->id, args);
        } else {
            builder.call_and_assign(out_var, "node_" + type->id, args);
        }

        param_offset += ctx.float_inputs.at(nid) + ctx.param_count.at(nid);
    }

    auto tail_it = ctx.node_type.find(group.nodes.back());
    const auto* tail_type = (tail_it != ctx.node_type.end()) ? tail_it->second : nullptr;
    bool tail_multi = tail_type && tail_type->outputs.size() > 1;

    if (tail_multi) {
        std::vector<uint32_t> slot_indices;
        for (uint32_t o = 0; o < tail_type->outputs.size(); ++o)
            slot_indices.push_back(o);
        builder.main_end_multi(slot_indices, [&](uint32_t i) {
            return tail_var + "_out" + std::to_string(i);
        });
    } else {
        builder.main_end(tail_var);
    }

    result.external_inputs = static_cast<uint32_t>(group.external_inputs.size());
    result.source = builder.build();
    return result;
}

} // namespace te::fusion
