#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/GlslBuilder.hpp"
#include "engine/PushConstants.hpp"
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace te {

namespace {

struct ExtSrc { uint32_t slot; };
struct RegSrc { size_t local_index; uint32_t output_socket = 0; };
struct ConstSrc { float value = 0.0f; };
using InputSrc = std::variant<ExtSrc, RegSrc, ConstSrc>;

struct NodeEmit {
    NodeId                  node_id = 0;
    std::string             type_id;
    std::vector<InputSrc>   input_srcs;
    uint32_t                param_count = 0;
    uint32_t                param_offset = 0;
    uint32_t                float_input_count = 0;
    uint32_t                output_count = 1;
    ChannelFormat           format_override = ChannelFormat::RGBA;
    bool                    is_format_sensitive = false;
};

using ConnByDst = std::unordered_map<
    NodeId,
    std::vector<std::pair<uint32_t, NodeId>>>;

ConnByDst index_connections(const GraphIR& ir) noexcept {
    ConnByDst out;
    for (const auto& c : ir.connections)
        out[c.dst_node].push_back({c.dst_socket, c.src_node});
    return out;
}

std::string local_var_name(size_t node_index, uint32_t output_socket, uint32_t output_count) {
    if (output_count <= 1)
        return "_local_" + std::to_string(node_index);
    return "_local_" + std::to_string(node_index) + "_" + std::to_string(output_socket);
}

} // namespace

FusedResult emit_fused_subgraph(
    const ActivePath& path,
    const GraphIR& ir,
    const NodeLibrary& lib,
    uint32_t chain_base_slot,
    const std::unordered_map<NodeId, int>& global_param_slots)
{
    FusedResult result;
    if (path.nodes.empty()) {
        result.error = "empty path";
        return result;
    }

    std::unordered_map<NodeId, size_t> node_to_index;
    node_to_index.reserve(path.nodes.size());
    for (size_t i = 0; i < path.nodes.size(); ++i)
        node_to_index[path.nodes[i]] = i;

    const ConnByDst conns = index_connections(ir);

    StorageFormat tail_sf{ChannelFormat::RGBA, BitDepth::F16};
    {
        NodeId tail_id = path.nodes.back();
        const auto* tail_inst = ir.find(tail_id);
        if (tail_inst) {
            tail_sf.channels = tail_inst->format_override;
            tail_sf.depth    = tail_inst->resolved_depth;
        }
    }

    glsl::GlslBuilder builder;
    builder.add_header(glsl::compute_header(tail_sf));
    builder.add_function(glsl::format_helpers());

    std::unordered_set<std::string> emitted_funcs;
    std::vector<NodeEmit> emits;
    uint32_t next_ext = 0;

    for (NodeId id : path.nodes) {
        const ValidatedNode* inst = ir.find(id);
        const NodeType* type = inst ? lib.find(inst->type_id) : nullptr;
        if (!inst || !type) {
            result.error = "node " + std::to_string(id) + " not found";
            return result;
        }

        NodeEmit ne;
        ne.node_id = id;
        ne.type_id = type->id;
        ne.param_count = static_cast<uint32_t>(type->params.size());
        // Use actual global SSBO slot offset relative to chain base.
        auto gslot_it = global_param_slots.find(id);
        ne.param_offset = (gslot_it != global_param_slots.end())
            ? gslot_it->second - chain_base_slot
            : 0;
        for (const auto& inp : type->inputs)
            if (inp.type == SocketType::Float) ++ne.float_input_count;
        ne.output_count = static_cast<uint32_t>(type->outputs.size());
        ne.format_override = inst->format_override;
        ne.is_format_sensitive = type->is_format_sensitive;

        auto it = conns.find(id);
        for (uint32_t s = 0; s < type->inputs.size(); ++s) {
            NodeId src = 0;
            if (it != conns.end()) {
                for (const auto& pr : it->second) {
                    if (pr.first == s) { src = pr.second; break; }
                }
            }
            if (src != 0 && node_to_index.count(src)) {
                uint32_t out_sock = 0;
                for (const auto& c : ir.connections) {
                    if (c.src_node == src && c.dst_node == id && c.dst_socket == s) {
                        out_sock = c.src_socket;
                        break;
                    }
                }
                ne.input_srcs.push_back(RegSrc{node_to_index.at(src), out_sock});
            } else if (src != 0) {
                // Connected to a node outside this chain group (split boundary).
                // Sample its output texture via bindless slot.
                ne.input_srcs.push_back(ExtSrc{next_ext++});
            } else {
                // Truly unconnected: bake default as GLSL constant.
                if (s < type->inputs.size() && type->inputs[s].type == SocketType::Vec4) {
                    ne.input_srcs.push_back(ConstSrc{type->inputs[s].default_value});
                } else if (s < type->inputs.size() && type->inputs[s].type == SocketType::Float) {
                    ne.input_srcs.push_back(ConstSrc{type->inputs[s].default_value});
                } else {
                    ne.input_srcs.push_back(ExtSrc{next_ext++});
                }
            }
        }

        emits.push_back(std::move(ne));

        // Build per-node external-socket mask for cache key.
        // Bit s = 1 means socket s of this node is an external input (ExtSrc).
        {
            const auto& ne_ref = emits.back();
            uint32_t mask = 0;
            for (uint32_t s = 0; s < ne_ref.input_srcs.size() && s < 32; ++s) {
                if (std::holds_alternative<ExtSrc>(ne_ref.input_srcs[s]))
                    mask |= (1u << s);
            }
            result.external_socket_masks.push_back(mask);
        }

        if (emitted_funcs.insert(type->glsl_function).second)
            builder.add_function(type->glsl_function);
    }

    if (next_ext > MAX_PASS_INPUTS) {
        result.error = "too many external inputs (" + std::to_string(next_ext) +
                       ", max " + std::to_string(MAX_PASS_INPUTS) + ")";
        return result;
    }
    result.external_inputs = next_ext;

    builder.main_begin();
    builder.statement("if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;");

    for (size_t i = 0; i < emits.size(); ++i) {
        const auto& ne = emits[i];
        const NodeType* type = lib.find(ne.type_id);
        if (!type) continue;

        // Declare external vec4 inputs (skip Sampler2D and ConstSrc).
        for (size_t s = 0; s < ne.input_srcs.size(); ++s) {
            if (std::holds_alternative<ConstSrc>(ne.input_srcs[s]))
                continue; // baked as GLSL constant, no slot needed
            if (std::holds_alternative<ExtSrc>(ne.input_srcs[s])) {
                if (s < type->inputs.size() && type->inputs[s].type == SocketType::Sampler2D)
                    continue; // Sampler2D handled inline via TSTexture constructor
                auto ext = std::get<ExtSrc>(ne.input_srcs[s]);
                builder.declare_external(
                    "_in_" + std::to_string(i) + "_" + std::to_string(s),
                    ext.slot);
            }
        }

        // Declare output locals.
        if (ne.output_count <= 1) {
            builder.declare_local("_local_" + std::to_string(i));
        } else {
            for (uint32_t o = 0; o < ne.output_count; ++o) {
                builder.declare_local("_local_" + std::to_string(i) + "_" + std::to_string(o));
            }
        }

        // Build function arguments.
        std::vector<std::string> args;
        args.push_back("uv");

        // Declared inputs.
        for (size_t s = 0; s < ne.input_srcs.size(); ++s) {
            bool is_float_input = (s < type->inputs.size() && type->inputs[s].type == SocketType::Float);
            bool is_sampler = (s < type->inputs.size() && type->inputs[s].type == SocketType::Sampler2D);

            if (std::holds_alternative<RegSrc>(ne.input_srcs[s])) {
                auto reg = std::get<RegSrc>(ne.input_srcs[s]);
                NodeId src_node = path.nodes[reg.local_index];
                const ValidatedNode* src_inst = ir.find(src_node);
                const NodeType* src_type = src_inst ? lib.find(src_inst->type_id) : nullptr;
                uint32_t src_out_count = src_type ? static_cast<uint32_t>(src_type->outputs.size()) : 1;
                std::string var = local_var_name(reg.local_index, reg.output_socket, src_out_count);
                if (is_float_input) var += ".r";
                args.push_back(std::move(var));
            } else if (std::holds_alternative<ConstSrc>(ne.input_srcs[s])) {
                auto c = std::get<ConstSrc>(ne.input_srcs[s]);
                if (is_float_input) {
                    uint32_t float_idx = 0;
                    for (uint32_t f = 0; f < s; ++f)
                        if (type->inputs[f].type == SocketType::Float) ++float_idx;
                    uint32_t ssbo_idx = ne.param_offset + ne.param_count + float_idx;
                    args.push_back("node_params[pc.param_ring_idx].v[pc.param_base_slot + " +
                                   std::to_string(ssbo_idx) + "]");
                } else {
                    args.push_back("vec4(" + std::to_string(c.value) + ")");
                }
            } else {
                auto ext = std::get<ExtSrc>(ne.input_srcs[s]);
                if (is_float_input) {
                    // Float input from outside chain group: sample texture, take .r.
                    args.push_back("_in_" + std::to_string(i) + "_" + std::to_string(s) + ".r");
                } else if (is_sampler) {
                    // Sampler2D: construct TSTexture with bindless slot + computed texel size.
                    uint32_t slot = ext.slot;
                    args.push_back("TSTexture(" + std::to_string(slot) +
                                   ", 1.0 / vec2(textureSize(u_sampled[nonuniformEXT("
                                   "pc.in_sampled_slots[" + std::to_string(slot) + "])], 0)))");
                } else {
                    args.push_back("_in_" + std::to_string(i) + "_" + std::to_string(s));
                }
            }
        }

        // Params.
        for (uint32_t p = 0; p < ne.param_count; ++p) {
            args.push_back("node_params[pc.param_ring_idx].v[pc.param_base_slot + " +
                           std::to_string(ne.param_offset) + " + " +
                           std::to_string(p) + "]");
        }

        // Emit the function call.
        if (ne.output_count <= 1) {
            builder.call_and_assign("_local_" + std::to_string(i),
                                    "node_" + type->id, args);
        } else {
            for (uint32_t o = 0; o < ne.output_count; ++o) {
                args.push_back("_local_" + std::to_string(i) + "_" + std::to_string(o));
            }
            builder.call_void("node_" + type->id, args);
        }

        // Format post-process IMMEDIATELY after each node so downstream
        // nodes in the chain see the formatted value (e.g. Mono zeroes G/B).
        if (ne.is_format_sensitive
            && ne.format_override != ChannelFormat::RGBA) {
            std::string var = (ne.output_count <= 1)
                ? "_local_" + std::to_string(i)
                : "_local_" + std::to_string(i) + "_0";
            static const char* fn_for[] = {
                "_fmt_mono", "_fmt_uv", "_fmt_rgb", "_fmt_rgba",
                "_fmt_id", "_fmt_metadata"
            };
            size_t fi = static_cast<size_t>(ne.format_override);
            const char* fn = (fi < std::size(fn_for)) ? fn_for[fi] : "_fmt_rgba";
            builder.statement(var + " = " + fn + "(" + var + ");");
        }
    }

    // Compute the tail variable name for imageStore output.
    const auto& tail = emits.back();
    std::string tail_var = (tail.output_count <= 1)
        ? "_local_" + std::to_string(emits.size() - 1)
        : "_local_" + std::to_string(emits.size() - 1) + "_0";

    builder.main_end(tail_var);
    result.source = builder.build();
    return result;
}

} // namespace te
