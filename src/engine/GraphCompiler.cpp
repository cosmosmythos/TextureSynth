#include "engine/GraphCompiler.hpp"
#include "engine/Graph.hpp"
#include "engine/PushConstants.hpp"
#include "engine/BindlessTable.hpp"
#include "engine/Logging.hpp"
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace te {

// Build a ShaderVariantKey for a node pass.
// Keyed by: (node_type, input_count, param_socket_mask, format_override).
//
// feature_flags layout (32 bits):
//   bits  0..2  : ChannelFormat enum (Mono=0, UV=1, RGB=2, RGBA=3, ID=4, Metadata=5)
//   bits  3..   : reserved for future flags (use_mask, hq, etc. — none defined yet)
//
// Why format_override is here: nodes with different output formats emit different
// GLSL (e.g. Perlin-mono returns vec4(n,n,n,1), Perlin-UV returns vec4(grad*0.5+0.5,0,1)).
// Keying on format means each variant gets its own .spv in ShaderCache, and the
// driver can specialize the math (no uniform branching).
static ShaderVariantKey build_variant_key(const NodeType& type,
                                           uint32_t input_count,
                                           uint32_t param_socket_count,
                                           ChannelFormat format) {

    ShaderVariantKey key;
    key.node_type_id   = type.id;
    key.input_count    = input_count;

    // Pack socket-driven params into a bitfield.
    // Bit i = 1 means param[i] is socket-driven (as_socket=true).
    uint32_t mask = 0;
    for (uint32_t i = 0; i < param_socket_count && i < 32; ++i) {
        if (i < type.params.size() && type.params[i].as_socket) {
            mask |= (1u << i);
        }
    }
    key.param_socket_mask = mask;

    // Pack format_override into the low 3 bits of feature_flags.
    // Cast through uint32_t so the static_cast below is well-defined.
    const uint32_t fmt_bits = static_cast<uint32_t>(format) & 0x7u;
    key.feature_flags = fmt_bits;

    return key;
}


std::string emit_node_shader(const ValidatedNode& vn,
                                    const NodeType& type,
                                    const ShaderVariantKey& key,
                                    int param_base,
                                    uint32_t input_count,
                                    ChannelFormat format,
                                    const std::vector<ResourceUUID>& input_resources) {
    (void)vn; (void)key; (void)param_base; (void)input_count;
    std::ostringstream s;

    s << "#version 460\n";
    // TS_FORMAT is consumed by the per-node format post-process (emitted
    // only for nodes flagged is_format_sensitive). The GLSL preprocessor
    // resolves the #if/#elif chain at parse time, so each cache key gets
    // its own dead-branch-stripped .spv.
    s << "#define TS_FORMAT " << static_cast<uint32_t>(format) << "\n";
    s << "#extension GL_EXT_nonuniform_qualifier : require\n";
    s << "#extension GL_EXT_samplerless_texture_functions : require\n";
    s << "layout(local_size_x = 8, local_size_y = 8) in;\n\n";

    // ── Bindless set 0 (forever-bound) ─────────────────────────────
    s << "layout(set = 0, binding = 0) uniform texture2D u_sampled[];\n";
    s << "layout(set = 0, binding = 1, rgba32f) writeonly uniform image2D u_storage[];\n";
    s << "layout(set = 0, binding = 2) uniform sampler samp_repeat;\n";
    s << "layout(set = 0, binding = 3) uniform sampler samp_clamp;\n";
    s << "layout(set = 0, binding = 4) uniform sampler samp_mirror;\n";
    s << "layout(set = 0, binding = 5, std430) readonly buffer NodeParams { float v[]; } "
    "node_params[" << BindlessTable::PARAM_RING_SIZE << "];\n\n";

    // ── Push constants ─────────────────────────────────────────────
    s << "layout(push_constant, std430) uniform PC {\n"
      << "    uint  resolution_x;\n"
      << "    uint  resolution_y;\n"
      << "    uint  seed;\n"
      << "    float time;\n"
      << "    uint  out_storage_slots[" << MAX_PASS_OUTPUTS << "];\n"
      << "    uint  param_base_slot;\n"
      << "    uint  input_count;\n"
      << "    uint  param_ring_idx;\n"
      << "    uint  in_sampled_slots[" << MAX_PASS_INPUTS << "];\n"
      << "} pc;\n\n";

    // ── Sampler2D helpers ──────────────────────────────────────────
    s << "struct TSTexture { int slot; vec2 inv_size; };\n\n";

    // CRITICAL: the runtime packs input_resources[] in socket-index order:
    //   indices [0 .. type.inputs.size())              -> declared inputs (any SocketType)
    //   indices [type.inputs.size() .. total_slots)    -> socket-driven params
    // So the local index for `type.inputs[i]` is exactly `i`.
    const uint32_t inputs_n = static_cast<uint32_t>(type.inputs.size());
    const uint32_t outputs_n = static_cast<uint32_t>(type.outputs.size());
    const bool multi_output = outputs_n > 1;

    bool any_sampler = false;
    for (const auto& sock : type.inputs)
        if (sock.type == SocketType::Sampler2D) { any_sampler = true; break; }

    if (any_sampler || inputs_n > 0) {
        s << "vec4 ts_sample(int local, vec2 uv) {\n"
          << "    uint gslot = pc.in_sampled_slots[local];\n"
          << "    return texture(sampler2D(u_sampled[nonuniformEXT(gslot)], samp_repeat), uv);\n"
          << "}\n"
          << "vec4 ts_sample_lod(int local, vec2 uv, float lod) {\n"
          << "    uint gslot = pc.in_sampled_slots[local];\n"
          << "    return textureLod(sampler2D(u_sampled[nonuniformEXT(gslot)], samp_repeat), uv, lod);\n"
          << "}\n"
          << "ivec2 ts_size(int local) {\n"
          << "    uint gslot = pc.in_sampled_slots[local];\n"
          << "    return textureSize(u_sampled[nonuniformEXT(gslot)], 0);\n"
          << "}\n"
          << "vec4  Sample      (TSTexture t, vec2 uv)            { return ts_sample(t.slot, uv); }\n"
          << "vec4  SampleLevel (TSTexture t, vec2 uv, float lod) { return ts_sample_lod(t.slot, uv, lod); }\n"
          << "ivec2 GetSize     (TSTexture t)                     { return ts_size(t.slot); }\n"
          << "vec2  GetTexelSize(TSTexture t)                     { return t.inv_size; }\n\n";

        // Per-Sampler2D-input handle constructor, indexed by *socket position*.
        for (uint32_t i = 0; i < inputs_n; ++i) {
            if (type.inputs[i].type != SocketType::Sampler2D) continue;
            s << "TSTexture ts_input_" << type.inputs[i].name << "() {\n"
              << "    int li = " << i << ";\n"
              << "    uint gslot = pc.in_sampled_slots[li];\n"
              << "    return TSTexture(li, 1.0 / vec2(textureSize(u_sampled[nonuniformEXT(gslot)], 0)));\n"
              << "}\n\n";
        }
    }

    // ── Node body ──────────────────────────────────────────────────
    s << type.glsl_function << "\n\n";

    s << "void main() {\n"
      << "    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);\n"
      << "    if (coord.x >= int(pc.resolution_x) || coord.y >= int(pc.resolution_y)) return;\n"
      << "    vec2 uv = vec2(coord) / vec2(max(pc.resolution_x - 1u, 1u), max(pc.resolution_y - 1u, 1u));\n";

    // Load non-sampler vec4/float inputs via bindless texelFetch, indexed by socket position.
    // Unconnected Vec4 inputs: baked as vec4(default_value) — no dummy texture.
    // Unconnected Float inputs: direct SSBO read — no sentinel branch.
    uint32_t float_input_idx = 0;
    for (uint32_t i = 0; i < inputs_n; ++i) {
        const auto& sock = type.inputs[i];
        if (sock.type == SocketType::Sampler2D) continue;
        if (sock.type == SocketType::Float) {
            uint32_t float_ssbo_idx = (uint32_t)type.params.size() + float_input_idx;
            bool connected = (i < input_resources.size() && input_resources[i].node_id != 0);
            if (connected) {
                s << "    float in" << i << " = texelFetch(u_sampled[nonuniformEXT(pc.in_sampled_slots["
                  << i << "])], coord, 0).r;\n";
            } else {
                s << "    float in" << i << " = node_params[pc.param_ring_idx].v[pc.param_base_slot + "
                  << float_ssbo_idx << "];\n";
            }
            ++float_input_idx;
        } else {
            bool connected = (i < input_resources.size() && input_resources[i].node_id != 0);
            if (connected) {
                s << "    vec4 in" << i << " = texelFetch(u_sampled[nonuniformEXT(pc.in_sampled_slots["
                  << i << "])], coord, 0);\n";
            } else {
                s << "    vec4 in" << i << " = vec4(" << sock.default_value << ");\n";
            }
        }
    }

    // Call node function: node_<id>(uv, inputs..., params...)
    if (multi_output) {
        s << "    vec4 out0, out1, out2, out3;\n";
        s << "    node_" << type.id << "(uv";
    }
    else {
        s << "    vec4 result = node_" << type.id << "(uv";
    }

    for (uint32_t i = 0; i < inputs_n; ++i) {
        s << ", ";
        if (type.inputs[i].type == SocketType::Sampler2D)
            s << "ts_input_" << type.inputs[i].name << "()";
        else
            s << "in" << i;
    }

    // Params: socket-driven ones live in in_sampled_slots[inputs_n + k] in the order they appear in type.params (matching GraphCompiler::compile()).
    uint32_t param_socket_local = inputs_n;
    for (size_t i = 0; i < type.params.size(); ++i) {
        s << ", ";
        if (type.params[i].as_socket) {
            // Hybrid: slider (SSBO) when unconnected (bindless slot < 6 = Mono dummy),
            // texture (.r) when connected (bindless slot >= 6 = real image).
            s << "(pc.in_sampled_slots[" << param_socket_local << "] < 6u"
              << " ? node_params[pc.param_ring_idx].v[pc.param_base_slot + " << i << "]"
              << " : texelFetch(u_sampled[nonuniformEXT(pc.in_sampled_slots["
              << param_socket_local << "])], coord, 0).r)";
            ++param_socket_local;
        } else {
            s << "node_params[pc.param_ring_idx].v[pc.param_base_slot + " << i << "]";
        }
    }
    if (multi_output) {
        for (uint32_t i = 0; i < outputs_n; ++i) {
            s << ", out" << i;
        }
    }
    s << ");\n";

    // ── Format post-process (noise-style outputs only) ──────────────
    // For nodes that return the canonical noise/gradient vec4(noise,
    // grad.x, grad.y, 1) we fold the channels into the requested output
    // format. Combiners and constant sources don't opt in — their `result`
    // is already a final color and must NOT be collapsed.
    //
    // The GLSL preprocessor resolves these #if branches at parse time, so
    // each ChannelFormat gets its own dead-stripped .spv in ShaderCache.
    if (type.is_format_sensitive && !multi_output) {
        s << "    // Format post-process (per .node.json format_sensitive)\n"
          << "#if TS_FORMAT == 0\n"        // Mono
          << "    result = vec4(result.r, result.r, result.r, 1.0);\n"
          << "#elif TS_FORMAT == 1\n"      // UV (use gradient, drop noise)
          << "    result = vec4(result.g, result.b, 0.0, 1.0);\n"
          << "#elif TS_FORMAT == 2\n"      // RGB
          << "    result = vec4(result.r, result.g, result.b, 1.0);\n"
          << "#elif TS_FORMAT == 3\n"      // RGBA
          << "    result = vec4(result.r, result.g, result.b, 1.0);\n"
          << "#else\n"                      // ID, Metadata — pass through
          << "    result = vec4(result.r, result.g, result.b, result.a);\n"
          << "#endif\n";
    }

    if (multi_output) {
        for (uint32_t i = 0; i < outputs_n; ++i)
            s << "    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots["
            << i << "])], coord, out" << i << ");\n";
    }
    else {
        s << "    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, result);\n";
    }
    s << "}\n";  // close main()
    return s.str();
}


CompileGraphResult GraphCompiler::compile(const GraphIR& ir, const NodeLibrary& lib) {

    CompileGraphResult result;
    if (ir.nodes.empty()) { result.error = "Graph has no nodes"; return result; }

    // 1. Assign SSBO base slots per node.
    // Layout per node: [manifest_params..., float_input_defaults...]
    std::unordered_map<NodeId, int> param_base_slot;
    int next_slot = 0;
    for (NodeId id : ir.eval_order) {
        auto* inst = ir.find(id);
        auto* type = lib.find(inst->type_id);
        param_base_slot[id] = next_slot;
        uint32_t float_input_count = 0;
        for (const auto& inp : type->inputs)
            if (inp.type == SocketType::Float) ++float_input_count;
        next_slot += (int)type->params.size() + (int)float_input_count;
    }

    // 2. Build the PassPlan with one pass per node, and emit GLSL per pass.
    PassPlan plan;
    plan.passes.reserve(ir.eval_order.size());

    for (NodeId id : ir.eval_order) {
        auto* inst = ir.find(id);
        auto* type = lib.find(inst->type_id);

        ComputePass pass;
        pass.node_id         = id;
        pass.type_id         = type->id;
		pass.output_resources.clear();
		for (uint32_t i = 0; i < type->outputs.size(); ++i) { pass.output_resources.push_back({ id, i }); }
        pass.param_base_slot = param_base_slot[id];
        pass.input_mode      = InputMode::PreSampled;
        // Phase 1c: mirror the bypassed flag from the validated node so
        // the executor can emit a clear-to-zero dispatch when set.
        pass.bypassed        = inst->bypassed;

        pass.kind            = inst->pass_kind;
    
        const uint32_t inputs_n = (uint32_t)type->inputs.size();
        uint32_t param_socket_count = 0;
        for (auto& p : type->params) if (p.as_socket) ++param_socket_count;
        const uint32_t total_slots = inputs_n + param_socket_count;
    
        pass.input_resources.assign(total_slots, ResourceUUID{});
        for (auto& c : ir.connections) {
            if (c.dst_node != id) continue;
            if (c.dst_socket < total_slots)
                pass.input_resources[c.dst_socket] = {c.src_node, c.src_socket};
        }
        pass.input_formats.clear();
        for (const auto& inp : type->inputs)
            pass.input_formats.push_back(inp.format);
    
        if (pass.kind == PassKind::Compute) {
            // Build variant key BEFORE emitting GLSL — cache lookup uses this.
            // format_override flows from ValidatedNode (set from NodeInstance
            // by validate_graph) into the cache key so each format gets its
            // own .spv. See build_variant_key for the encoding.
            pass.variant_key = build_variant_key(*type, total_slots, param_socket_count, inst->format_override);
            pass.shader_glsl = emit_node_shader(*inst, *type, pass.variant_key, pass.param_base_slot, total_slots, inst->format_override, pass.input_resources);
        }
        pass.input_socket_count = total_slots;
        plan.passes.push_back(std::move(pass));
    }
    plan.final_output_resource = {ir.output_node, ir.output_socket};

    // Stage 6 aliasing: compute lifetimes and color classes (no chains).
    {
        plan.chain_index_of_pass.assign(plan.passes.size(), UINT32_MAX);

        for (uint32_t i = 0; i < (uint32_t)plan.passes.size(); ++i) {
            const auto& pass = plan.passes[i];
            for (const auto& rid : pass.output_resources) {
                auto& lt = plan.lifetimes[rid];
                if (lt.first_pass == UINT32_MAX) lt.first_pass = i;
                lt.last_pass = i;
            }
            for (const auto& rid : pass.input_resources) {
                if (rid.node_id == 0) continue;
                auto& lt = plan.lifetimes[rid];
                lt.last_pass = i;
                if (lt.first_pass == UINT32_MAX) lt.first_pass = 0;
            }
        }
        auto& fo_lt = plan.lifetimes[plan.final_output_resource];
        fo_lt.first_pass = 0;
        fo_lt.last_pass = UINT32_MAX;

        struct Colored { ResourceUUID rid; uint32_t first, last; };
        std::vector<Colored> items;
        for (auto& kv : plan.lifetimes) {
            if (kv.first == plan.final_output_resource) continue;
            if (kv.second.first_pass == kv.second.last_pass) continue;
            if (kv.second.last_pass == UINT32_MAX) continue;
            items.push_back({kv.first, kv.second.first_pass, kv.second.last_pass});
        }
        std::sort(items.begin(), items.end(),
            [](const Colored& a, const Colored& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.last < b.last;
            });

        std::vector<uint32_t> color_end;
        uint32_t next_color = 1;
        for (auto& item : items) {
            bool found = false;
            for (uint32_t c = 0; c < (uint32_t)color_end.size(); ++c) {
                if (color_end[c] < item.first) {
                    plan.color_classes[item.rid] = c + 1;
                    color_end[c] = item.last;
                    found = true;
                    break;
                }
            }
            if (!found) {
                plan.color_classes[item.rid] = next_color;
                color_end.push_back(item.last);
                ++next_color;
            }
        }
    }

    // ── User-facing error FIRST (param budget) ─────────────────────
    // next_slot is the total floats consumed by the graph. The legal
    // range is [0, MAX_NODE_PARAMS] inclusive — strictly greater means
    // the last param would write past the SSBO end.
    if (next_slot > static_cast<int>(MAX_NODE_PARAMS)) {
        result.success = false;
        result.error = "Param budget exceeded: graph needs "
                     + std::to_string(next_slot) + " floats, max is "
                     + std::to_string(MAX_NODE_PARAMS)
                     + ". Split the graph or raise MAX_NODE_PARAMS in Graph.hpp.";
        return result;
    }

    // ── Internal invariants (should never fire; bugs if they do) ───
    if (plan.passes.size() != ir.nodes.size()) {
        result.success = false;
        result.error = "internal: pass count != node count";
        return result;
    }
    std::unordered_set<ResourceUUID, ResourceUUIDHash> seen;
    for (auto& p : plan.passes) {
        for (auto& rid : p.output_resources) {
            if (!seen.insert(rid).second) {
                result.error = "internal: duplicate output_resource for node "
                    + std::to_string(rid.node_id)
                    + " output " + std::to_string(rid.output_index);
                return result;
            }
        }
    }

    plan.active_resources.insert(plan.final_output_resource);
    for (const auto& pass : plan.passes) {
        for (const auto& rid : pass.output_resources) {
            plan.active_resources.insert(rid);
        }
    }

    result.success            = true;
    result.pass_plan          = std::move(plan);
    result.param_base_slot    = std::move(param_base_slot);
    result.total_param_floats = next_slot;
    return result;
}

} // namespace te