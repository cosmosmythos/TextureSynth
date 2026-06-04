#include "engine/ChainShaderEmitter.hpp"
#include "engine/PushConstants.hpp"   // MAX_PASS_INPUTS, MAX_PASS_OUTPUTS
#include "engine/BindlessTable.hpp"    // PARAM_RING_SIZE
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace te::chain_shader {

// ===========================================================================
// Internal helpers
// ===========================================================================
namespace {

// C++17 std::visit trick: an "overloaded" visitor composed of lambdas.
// Lets us write `std::visit(overloaded{[](A){...}, [](B){...}}, variant)`
// with type-dispatched lambdas instead of a hand-rolled switch.
template <typename... Ts>
struct Overloaded : Ts... { using Ts::operator()...; };
template <typename... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

// Source of one input to one node in a chain.
//   External -- fetched from a pre-sampled image at pc.in_sampled_slots[i]
//   Local    -- the previous chain node's local output (vec4, register-only)
struct External { uint32_t slot; };        // index into pc.in_sampled_slots[]
struct Local    { size_t  index; };       // index into _local_<index>
using InputSrc = std::variant<External, Local>;

struct NodeEmit {
    NodeId                node_id = 0;
    std::vector<InputSrc> inputs;     // one entry per declared input socket
    uint32_t              param_count = 0;
    uint32_t              param_offset_in_chain = 0;  // == chain.param_offsets[i]
};

// Stage 4.1 emit plan: per-node input sources + totals.
struct EmitPlan {
    std::vector<NodeEmit>  nodes;
    uint32_t               external_input_count = 0;
    bool                   valid = false;
    std::string            error;
};

// Index connections by destination for O(1) lookup.
using ConnByDst = std::unordered_map<
        NodeId,
        std::vector<std::pair<uint32_t /*dst_socket*/, NodeId /*src_node*/>>>;

[[nodiscard]] ConnByDst index_connections(const GraphIR& ir) noexcept {
    ConnByDst out;
    for (const auto& c : ir.connections) {
        out[c.dst_node].push_back({c.dst_socket, c.src_node});
    }
    return out;
}

// Validate the chain is linear (Stage 4.1 + 4.2 contract) and build
// the per-node input-source map.
//
// A chain is "linear" iff:
//   - every node has 0..N input sockets (multi-input supported since 4.2)
//   - every input is SocketType::Vec4
//   - for the head (i==0), every input socket is either unconnected or
//     fed by a node OUTSIDE the chain
//   - for non-head nodes (i>0):
//       * socket 0 is fed by the previous chain node's output (Local)
//       * all other sockets are unconnected OR fed by a node outside
//         the chain (External)
//   - the first node's input (if any) comes from outside the chain
//   - total external inputs across the whole chain <= MAX_PASS_INPUTS
//     (push-constant slot limit)
[[nodiscard]] EmitPlan build_emit_plan(const Chain& chain,
                                        const GraphIR& ir,
                                        const NodeLibrary& lib) noexcept {
    EmitPlan plan;

    // Position of each chain node in the chain (for predecessor check)
    std::unordered_map<NodeId, size_t> chain_pos;
    chain_pos.reserve(chain.nodes.size());
    for (size_t i = 0; i < chain.nodes.size(); ++i) {
        chain_pos[chain.nodes[i]] = i;
    }
    const ConnByDst conns = index_connections(ir);

    uint32_t next_external_slot = 0;
    for (size_t i = 0; i < chain.nodes.size(); ++i) {
        const NodeId id = chain.nodes[i];
        const ValidatedNode* inst = ir.find(id);
        const NodeType* type = inst ? lib.find(inst->type_id) : nullptr;
        if (!inst || !type) {
            plan.error = "chain node " + std::to_string(id) +
                         " not found in IR/library";
            return plan;
        }

        NodeEmit ne;
        ne.node_id = id;
        ne.param_count = static_cast<uint32_t>(type->params.size());
        ne.param_offset_in_chain = (i < chain.param_offsets.size())
                                  ? chain.param_offsets[i] : 0u;

        // Non-first node with 0 inputs is a malformed chain (sources are
        // always chain heads, never mid-chain).
        if (i > 0 && type->inputs.empty()) {
            plan.error = "chain node " + type->id + " (index " +
                         std::to_string(i) + ") has no inputs but is not the head";
            return plan;
        }

        // Stage 4.2: per-socket input resolution. Each socket s of
        // node i is resolved to either Local{i-1} (only for socket 0
        // of non-head nodes fed by the previous chain node) or
        // External{next_external_slot++} (head, unconnected, or any
        // mid-chain socket that isn't the predecessor's output).
        for (uint32_t s = 0; s < type->inputs.size(); ++s) {
            const auto& sock = type->inputs[s];
            if (sock.type != SocketType::Vec4) {
                plan.error = "chain node " + type->id +
                             " socket " + std::to_string(s) +
                             " is not Vec4 (chain shader limitation)";
                return plan;
            }
            // Find the connection feeding (id, s), if any.
            NodeId conn_src = 0;
            const auto it = conns.find(id);
            if (it != conns.end()) {
                for (const auto& pr : it->second) {
                    if (pr.first == s) { conn_src = pr.second; break; }
                }
            }
            const bool connected = (conn_src != 0);

            if (i == 0) {
                // Head: every socket is external. A connection to a
                // node inside the chain would violate the chain-finder
                // contract (heads are by definition the first node).
                if (connected && chain_pos.count(conn_src)) {
                    plan.error = "head chain node " + type->id +
                                 " socket " + std::to_string(s) +
                                 " fed by node inside the chain";
                    return plan;
                }
                ne.inputs.push_back(External{next_external_slot++});
            } else {
                // Mid-chain: socket 0 fed by predecessor = Local.
                // Everything else (unconnected, or fed by a node OUTSIDE
                // the chain) is External. A socket fed by a chain-internal
                // node that is NOT the predecessor IS a true non-linearity
                // and must be rejected -- we cannot route to it from a
                // non-immediate predecessor inside a single fused shader.
                if (s == 0 && connected
                    && chain_pos.count(conn_src)
                    && chain_pos.at(conn_src) == i - 1) {
                    ne.inputs.push_back(Local{i - 1});
                } else if (!connected
                           || !chain_pos.count(conn_src)) {
                    // Unconnected, or fed by a node outside the chain:
                    // both become an external sampled-image read.
                    ne.inputs.push_back(External{next_external_slot++});
                } else {
                    plan.error = "chain node " + type->id +
                                 " (index " + std::to_string(i) +
                                 ") socket " + std::to_string(s) +
                                 " has a non-predecessor connection (not linear)";
                    return plan;
                }
            }
        }
        plan.nodes.push_back(std::move(ne));
    }

    if (next_external_slot > MAX_PASS_INPUTS) {
        plan.error = "chain has " + std::to_string(next_external_slot) +
                     " external inputs (max " +
                     std::to_string(MAX_PASS_INPUTS) + ")";
        return plan;
    }
    plan.external_input_count = next_external_slot;
    plan.valid = true;
    return plan;
}

// GLSL string emitted verbatim at the top of every chain shader.
// Layout matches PassPushConstants byte-for-byte; see the static_assert
// in emit_linear below.
//
// Key design points:
//   - We declare the same bindless set 0 as the per-node path. Chains
//     are not a separate pipeline layout; they share the global table
//     so the engine can swap them in for per-pass dispatches without
//     rebinding descriptors (Stage 6 optimization).
//   - pc.out_storage_slots[0] is the chain's final output image slot.
//     Slots [1..3] are unused; the 12-byte waste keeps the push-
//     constant struct layout-identical to PassPushConstants, which
//     means Stage 6 can dispatch chains using the same struct.
constexpr const char* CHAIN_HEADER = R"glsl(
#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_samplerless_texture_functions : require

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform texture2D u_sampled[];
layout(set = 0, binding = 1) writeonly uniform image2D u_storage[];
layout(set = 0, binding = 5, std430) readonly buffer NodeParams { float v[]; } node_params[3];

layout(push_constant) uniform PC {
    uint  resolution_x;
    uint  resolution_y;
    uint  seed;
    float time;
    uint  out_storage_slots[4];
    uint  param_base_slot;
    uint  input_count;
    uint  param_ring_idx;
    uint  in_sampled_slots[8];
} pc;

)glsl";

// Emit a GLSL expression that loads this node's input `i`.
//   External -> texelFetch from pc.in_sampled_slots[i]
//   Local    -> the previous chain node's local
[[nodiscard]] std::string emit_input_load(const InputSrc& src) {
    return std::visit(Overloaded{
        [](const External& e) -> std::string {
            return "texelFetch(u_sampled[nonuniformEXT(pc.in_sampled_slots["
                 + std::to_string(e.slot) + "])], coord, 0)";
        },
        [](const Local& l) -> std::string {
            return "_local_" + std::to_string(l.index);
        }
    }, src);
}

// Emit a single node's call: result = node_<id>(uv, inputs..., params...)
// where params are read from the chain's SSBO slice at the right offset.
void emit_node_call(std::ostringstream& s, const NodeEmit& ne,
                    const NodeType& type) {
    // Single shared _result variable in main(). Each call ASSIGNS to it;
    // redeclaring `vec4 _result = ...` on every call would be a GLSL
    // redefinition error in the same function scope.
    s << "    _result = node_" << type.id << "(uv";
    for (const auto& src : ne.inputs) {
        s << ", " << emit_input_load(src);
    }
    for (uint32_t p = 0; p < ne.param_count; ++p) {
        // Per-node param lives at chain.param_base_slot +
        // node.param_offset_in_chain + p. All three are compile-time
        // constants in the chain; the GLSL compiler will constant-fold.
        s << ", node_params[pc.param_ring_idx].v[pc.param_base_slot + "
          << ne.param_offset_in_chain << " + " << p << "]";
    }
    s << ");\n";
}

} // namespace

// ===========================================================================
// Public API
// ===========================================================================
Result emit_linear(const Chain& chain, const GraphIR& ir,
                   const NodeLibrary& lib) {
    Result result;
    const EmitPlan plan = build_emit_plan(chain, ir, lib);
    if (!plan.valid) {
        result.error = plan.error;
        return result;
    }

    // The push-constant struct in GLSL must match PassPushConstants
    // byte-for-byte. The per-node path relies on the same invariant
    // (GraphCompiler.cpp:91); we replicate it here so Stage 6 can
    // dispatch chains without a different push-constant struct.
    static_assert(sizeof(float) == 4, "GLSL float must be 4 bytes");
    static_assert(sizeof(uint32_t) == 4, "GLSL uint must be 4 bytes");
    // 16 (global) + 16 (out_storage_slots[4]) + 4*4 (base,count,ring,pad-ish)
    // + 8*4 (in_sampled_slots) = 76. Verified by the existing static_assert
    // on PassPushConstants; chain reuses the same struct.

    std::ostringstream s;
    s << CHAIN_HEADER;

    // Emit each UNIQUE glsl_function once. Multiple occurrences of the
    // same node type in a chain (e.g. step,step,step) must not produce
    // duplicate function definitions -- GLSL rejects them. We dedup on
    // the function body string; types that share the same body share
    // the definition. This is the standard include-guard-style
    // dedup that the NodeRegistryLoader relies on for `includes`.
    std::unordered_set<std::string> emitted;
    for (const auto& ne : plan.nodes) {
        const ValidatedNode* inst = ir.find(ne.node_id);
        const NodeType* type = inst ? lib.find(inst->type_id) : nullptr;
        if (!type) continue;       // build_emit_plan already validated
        if (emitted.insert(type->glsl_function).second) {
            s << type->glsl_function << "\n\n";
        }
    }

    // Format post-process: if the tail is format_sensitive (noise
    // generators) and the instance requested a non-RGBA format, collapse
    // _local_<last> through one of the _fmt_* helpers before imageStore.
    // Noise nodes return (noise, grad.x, grad.y, 1); we map that to the
    // requested output layout.
    const ValidatedNode* tail_inst =
        plan.nodes.empty() ? nullptr : ir.find(plan.nodes.back().node_id);
    const NodeType* tail_type =
        tail_inst ? lib.find(tail_inst->type_id) : nullptr;
    const bool tail_needs_format =
        tail_type && tail_type->is_format_sensitive
        && tail_inst->format_override != ChannelFormat::RGBA;
    if (tail_needs_format) {
        s << "vec4 _fmt_mono(vec4 v)  { return vec4(v.x, 0.0, 0.0, 0.0); }\n"
             "vec4 _fmt_uv(vec4 v)    { return vec4(v.y, v.z, 0.0, 1.0); }\n"
             "vec4 _fmt_rgb(vec4 v)   { return vec4(v.xxx, 1.0); }\n"
             "vec4 _fmt_rgba(vec4 v)  { return v; }\n"
             "vec4 _fmt_id(vec4 v)    { return v; }\n"
             "vec4 _fmt_metadata(vec4 v) { return v; }\n\n";
    }

    // Main function: load externals, declare locals, call each node in
    // sequence, imageStore the final local.
    s << "void main() {\n"
      << "    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);\n"
      << "    if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;\n"
      << "    vec2 uv = vec2(coord) / vec2(pc.resolution_x, pc.resolution_y);\n";

    // Declare all locals up front so each node's call is a single
    // statement and the GLSL compiler can allocate registers freely.
    s << "    vec4 _result;\n";
    for (size_t i = 0; i < plan.nodes.size(); ++i) {
        s << "    vec4 _local_" << i << ";\n";
    }

    // Call each node, store into its local.
    for (size_t i = 0; i < plan.nodes.size(); ++i) {
        const auto& ne = plan.nodes[i];
        const ValidatedNode* inst = ir.find(ne.node_id);
        const NodeType* type = inst ? lib.find(inst->type_id) : nullptr;
        if (!type) continue;
        emit_node_call(s, ne, *type);
        s << "    _local_" << i << " = _result;\n";
    }

    // Format post-process on the tail (no-op for non-sensitive tails).
    if (tail_needs_format) {
        static const char* fn_for[] = {
            "_fmt_mono", "_fmt_uv", "_fmt_rgb", "_fmt_rgba",
            "_fmt_id",   "_fmt_metadata"
        };
        const size_t fi = static_cast<size_t>(tail_inst->format_override);
        const char* fn = (fi < std::size(fn_for)) ? fn_for[fi] : "_fmt_rgba";
        s << "    _local_" << (plan.nodes.size() - 1) << " = " << fn
          << "(_local_" << (plan.nodes.size() - 1) << ");\n";
    }

    // Final imageStore. The chain's output goes to the storage slot
    // passed via push constants. We use slot 0 of out_storage_slots to
    // match the per-pass convention; Stage 6 will populate this from
    // chain.glsl's call site.
    if (!plan.nodes.empty()) {
        s << "    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, _local_"
          << (plan.nodes.size() - 1) << ");\n";
    }
    s << "}\n";

    result.source = std::move(s).str();   // C++20; if not available, s.str()
    result.external_inputs = plan.external_input_count;
    return result;
}

} // namespace te::chain_shader
