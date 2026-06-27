#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/GlslBuilder.hpp"
#include "engine/PushConstants.hpp"
#include <array>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace te {

namespace {

struct ExtSrc { uint32_t slot; };
struct RegSrc { size_t local_index; uint32_t output_socket = 0; };
struct ConstSrc { std::array<float, 4> value = {0.0f, 0.0f, 0.0f, 0.0f}; };
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
};

using ConnByDst = std::unordered_map<NodeId, std::vector<std::pair<uint32_t, NodeId>>>;

ConnByDst index_connections(const GraphIR& ir) noexcept {
    ConnByDst out;
    for (const auto& c : ir.connections)
        out[c.dst_node].push_back({c.dst_socket, c.src_node});
    return out;
}

std::string local_var_name_fallback(size_t node_index, uint32_t output_socket, uint32_t output_count) {
    if (output_count <= 1)
        return "_local_" + std::to_string(node_index);
    return "_local_" + std::to_string(node_index) + "_" + std::to_string(output_socket);
}

std::string colored_var_name(
    const ActivePath& path,
    size_t node_index,
    uint32_t output_socket,
    uint32_t output_count,
    const register_allocation::ColoringResult* coloring)
{
    if (!coloring) return local_var_name_fallback(node_index, output_socket, output_count);

    NodeId node_id = path.nodes[node_index];
    ResourceUUID rid{node_id, output_socket};
    auto it = coloring->assignment.find(rid);
    if (it != coloring->assignment.end()) {
        return "r" + std::to_string(it->second);
    }
    // Fallback for uncolored (external inputs, constants, etc).
    return local_var_name_fallback(node_index, output_socket, output_count);
}

} // namespace

FusedResult emit_fused_subgraph(
    const ActivePath& path,
    const GraphIR& ir,
    const NodeLibrary& lib,
    uint32_t chain_base_slot,
    const std::unordered_map<NodeId, int>& global_param_slots,
    const register_allocation::ColoringResult* coloring,
    const std::unordered_set<ResourceUUID, ResourceUUIDHash>* active_resources)
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
            tail_sf.channels = resolve_node_storage(*tail_inst, lib).channels;
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

        auto it = conns.find(id);
        for (uint32_t s = 0; s < type->inputs.size(); ++s) {
            NodeId src = 0;
            if (it != conns.end()) {
                for (const auto& pr : it->second) {
                    if (pr.first == s) { src = pr.second; break; }
                }
            }
            bool is_sampler_input = (s < type->inputs.size() &&
                                     type->inputs[s].type == SocketType::Sampler2D);
            if (src != 0 && node_to_index.count(src) && !is_sampler_input) {
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
                // Vec4 -> default_vec4 baked as vec4(r,g,b,a).
                // Float -> payload unused (SSBO read at runtime), value goes to [0] for symmetry.
                if (s < type->inputs.size() && type->inputs[s].type == SocketType::Vec4) {
                    ne.input_srcs.push_back(ConstSrc{type->inputs[s].default_vec4});
                } else if (s < type->inputs.size() && type->inputs[s].type == SocketType::Float) {
                    ne.input_srcs.push_back(ConstSrc{{type->inputs[s].default_value, 0.0f, 0.0f, 0.0f}});
                } else {
                    ne.input_srcs.push_back(ExtSrc{next_ext++});
                }
            }
        }

        emits.push_back(std::move(ne));

        // Build per-node external-socket mask + per-socket internal-producer index.
        // Both feed the FusedVariantKey so swapped internal wiring can't collide.
        {
            const auto& ne_ref = emits.back();
            uint32_t mask = 0;
            for (uint32_t s = 0; s < ne_ref.input_srcs.size(); ++s) {
                const auto& src = ne_ref.input_srcs[s];
                if (std::holds_alternative<ExtSrc>(src)) {
                    if (s < 32) mask |= (1u << s);
                    result.internal_producer_indices.push_back(UINT32_MAX);
                } else if (std::holds_alternative<RegSrc>(src)) {
                    result.internal_producer_indices.push_back(
                        static_cast<uint32_t>(std::get<RegSrc>(src).local_index));
                } else {
                    result.internal_producer_indices.push_back(UINT32_MAX);
                }
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

    // Declare shared memory pool for spilled variables.
    if (coloring && coloring->shared_slot_count > 0) {
        builder.declare_shared(coloring->shared_slot_count);
    }

    // Track which variables have been declared to avoid redeclaring reused registers.
    std::unordered_set<std::string> declared_vars;

    // Track which spilled resources have been loaded into temps.
    std::unordered_map<ResourceUUID, std::string, ResourceUUIDHash> spill_temps;

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

        // Declare output locals (skip if already declared — register reuse).
        if (ne.output_count <= 1) {
            std::string var = colored_var_name(path, i, 0, 1, coloring);
            if (declared_vars.insert(var).second)
                builder.declare_local(var);
        } else {
            for (uint32_t o = 0; o < ne.output_count; ++o) {
                std::string var = colored_var_name(path, i, o, ne.output_count, coloring);
                if (declared_vars.insert(var).second)
                    builder.declare_local(var);
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
                ResourceUUID src_rid{src_node, reg.output_socket};
                // Check if source is spilled — load from shared memory.
                if (coloring && coloring->spilled_assignment.count(src_rid)) {
                    uint32_t slot = coloring->spilled_assignment.at(src_rid);
                    auto it = spill_temps.find(src_rid);
                    if (it == spill_temps.end()) {
                        std::string temp = "_spill_" + std::to_string(slot);
                        builder.statement("vec4 " + temp + " = " +
                                          builder.spill_load_expr(slot) + ";");
                        spill_temps[src_rid] = temp;
                        it = spill_temps.find(src_rid);
                    }
                    std::string var = it->second;
                    if (is_float_input) var += ".r";
                    args.push_back(std::move(var));
                } else {
                    std::string var = colored_var_name(path, reg.local_index, reg.output_socket, src_out_count, coloring);
                    if (is_float_input) var += ".r";
                    args.push_back(std::move(var));
                }
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
                    args.push_back("vec4(" + std::to_string(c.value[0]) + ", "
                                   + std::to_string(c.value[1]) + ", "
                                   + std::to_string(c.value[2]) + ", "
                                   + std::to_string(c.value[3]) + ")");
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
            builder.call_and_assign(colored_var_name(path, i, 0, 1, coloring),
                                    "node_" + type->id, args);
            // Spill store: if this output is spilled, write to shared memory.
            if (coloring) {
                ResourceUUID rid{path.nodes[i], 0};
                auto sit = coloring->spilled_assignment.find(rid);
                if (sit != coloring->spilled_assignment.end()) {
                    builder.spill_store(sit->second, colored_var_name(path, i, 0, 1, coloring));
                }
            }
        } else {
            for (uint32_t o = 0; o < ne.output_count; ++o) {
                args.push_back(colored_var_name(path, i, o, ne.output_count, coloring));
            }
            builder.call_void("node_" + type->id, args);
            // Spill store for multi-output nodes.
            if (coloring) {
                for (uint32_t o = 0; o < ne.output_count; ++o) {
                    ResourceUUID rid{path.nodes[i], o};
                    auto sit = coloring->spilled_assignment.find(rid);
                    if (sit != coloring->spilled_assignment.end()) {
                        builder.spill_store(sit->second,
                            colored_var_name(path, i, o, ne.output_count, coloring));
                    }
                }
            }
        }

        // Format post-process IMMEDIATELY after each node so downstream
        // nodes in the chain see the formatted value (e.g. Mono zeroes G/B).
        if (ne.format_override != ChannelFormat::RGBA) {
            std::string var = (ne.output_count <= 1)
                ? colored_var_name(path, i, 0, 1, coloring)
                : colored_var_name(path, i, 0, ne.output_count, coloring);
            static const char* fn_for[] = {
                "_fmt_mono", "_fmt_uv", "_fmt_rgb", "_fmt_rgba"
            };
            size_t fi = static_cast<size_t>(ne.format_override);
            const char* fn = (fi < std::size(fn_for)) ? fn_for[fi] : "_fmt_rgba";
            builder.statement(var + " = " + fn + "(" + var + ");");
        }
    }

    // Compute the tail variable name for imageStore output.
    const auto& tail = emits.back();
    if (tail.output_count <= 1) {
        std::string tail_var = colored_var_name(path, emits.size() - 1, 0, 1, coloring);
        builder.main_end(tail_var);
    } else {
        // Multi-output tail: only write outputs that are in active_resources.
        size_t tail_idx = emits.size() - 1;
        uint32_t oc = tail.output_count;
        NodeId tail_node = path.nodes[tail_idx];
        std::vector<uint32_t> materialized;
        for (uint32_t o = 0; o < oc; ++o) {
            ResourceUUID rid{tail_node, o};
            if (!active_resources || active_resources->count(rid))
                materialized.push_back(o);
        }
        if (materialized.size() == 1 && materialized[0] == 0) {
            std::string tail_var = colored_var_name(path, tail_idx, 0, 1, coloring);
            builder.main_end(tail_var);
        } else if (materialized.empty()) {
            builder.main_end("vec4(0.0)");
        } else {
            builder.main_end_multi(materialized,
                [&, tail_idx, oc](uint32_t mi) -> std::string {
                    return colored_var_name(path, tail_idx, materialized[mi], oc, coloring);
                });
        }
    }
    result.source = builder.build();
    return result;
}

} // namespace te
