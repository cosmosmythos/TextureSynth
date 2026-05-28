#include "engine/GraphCompiler.hpp"
#include "engine/Graph.hpp"
#include "engine/PushConstants.hpp"
#include "engine/BindlessTable.hpp"
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace te {

// Build a ShaderVariantKey for a node pass.
// Today: keyed by (node_type, input_count, socket_param_mask, output_format).
// Tomorrow: feature_flags will be populated from node params (e.g. "use_mask").
static ShaderVariantKey build_variant_key(const NodeType& type,
                                          uint32_t input_count,
                                          uint32_t param_socket_count,
                                          uint32_t output_format) {
    ShaderVariantKey key;
    key.node_type_id   = type.id;
    key.input_count    = input_count;
    key.output_format  = output_format;

    // Pack socket-driven params into a bitfield.
    // Bit i = 1 means param[i] is socket-driven (as_socket=true).
    uint32_t mask = 0;
    for (uint32_t i = 0; i < param_socket_count && i < 32; ++i) {
        if (i < type.params.size() && type.params[i].as_socket) {
            mask |= (1u << i);
        }
    }
    key.param_socket_mask = mask;

    // feature_flags: today always 0. Retrofit path:
    //   if (node.param("use_mask") > 0.5f) key.feature_flags |= (1u << 0);
    //   if (node.param("high_quality") > 0.5f) key.feature_flags |= (1u << 1);

    return key;
}


static const char* format_qualifier_for(uint32_t vk_format) {
    // Mirrors Engine::set_precision mapping.
    switch (vk_format) {
        case VK_FORMAT_R8G8B8A8_UNORM:      return "rgba8";
        case VK_FORMAT_R16G16B16A16_SFLOAT: return "rgba16f";
        case VK_FORMAT_R32G32B32A32_SFLOAT: return "rgba32f";
        default:                            return "rgba32f";
    }
}

static std::string emit_node_shader(const ValidatedNode& vn,
                                    const NodeType& type,
                                    const ShaderVariantKey& key,
                                    int param_base,
                                    uint32_t input_count) {
    (void)vn; (void)param_base; (void)input_count;
    std::ostringstream s;

    s << "#version 460\n";
    s << "#extension GL_EXT_nonuniform_qualifier : require\n";
    s << "#extension GL_EXT_samplerless_texture_functions : require\n";
    s << "layout(local_size_x = 8, local_size_y = 8) in;\n\n";

    // ── Bindless set 0 (forever-bound) ─────────────────────────────
    const char* fmt_q = format_qualifier_for(key.output_format);
    s << "layout(set = 0, binding = 0) uniform texture2D u_sampled[];\n";
    s << "layout(set = 0, binding = 1, " << fmt_q << ") uniform image2D u_storage[];\n";
    s << "layout(set = 0, binding = 2) uniform sampler samp_repeat;\n";
    s << "layout(set = 0, binding = 3) uniform sampler samp_clamp;\n";
    s << "layout(set = 0, binding = 4) uniform sampler samp_mirror;\n";
    s << "layout(set = 0, binding = 5, std430) readonly buffer NodeParams { float v[]; } "
    "node_params[" << BindlessTable::PARAM_RING_SIZE << "];\n\n";

    // ── Push constants — MUST mirror PassPushConstants (64 bytes) ──
    s << "layout(push_constant) uniform PC {\n"
      << "    uint  resolution_x;\n"
      << "    uint  resolution_y;\n"
      << "    uint  seed;\n"
      << "    float time;\n"
      << "    uint  out_storage_slot;\n"
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
      << "    vec2 uv = vec2(coord) / vec2(pc.resolution_x, pc.resolution_y);\n";

    // Load non-sampler vec4/float inputs via bindless texelFetch, indexed by socket position.
    for (uint32_t i = 0; i < inputs_n; ++i) {
        const auto& sock = type.inputs[i];
        if (sock.type == SocketType::Sampler2D) continue;
        s << "    vec4 in" << i << " = texelFetch(u_sampled[nonuniformEXT(pc.in_sampled_slots["
          << i << "])], coord, 0);\n";
    }

    // Call node function: node_<id>(uv, inputs..., params...)
    s << "    vec4 result = node_" << type.id << "(uv";
    for (uint32_t i = 0; i < inputs_n; ++i) {
        s << ", ";
        if (type.inputs[i].type == SocketType::Sampler2D)
            s << "ts_input_" << type.inputs[i].name << "()";
        else
            s << "in" << i;
    }

    // Params: socket-driven ones live in in_sampled_slots[inputs_n + k] in
    // the order they appear in type.params (matching GraphCompiler::compile()).
    uint32_t param_socket_local = inputs_n;
    for (size_t i = 0; i < type.params.size(); ++i) {
        s << ", ";
        if (type.params[i].as_socket) {
            s << "texelFetch(u_sampled[nonuniformEXT(pc.in_sampled_slots["
              << param_socket_local << "])], coord, 0).r";
            ++param_socket_local;
        } else {
            s << "node_params[pc.param_ring_idx].v[pc.param_base_slot + " << i << "]";
        }
    }
    s << ");\n";

    s << "    imageStore(u_storage[nonuniformEXT(pc.out_storage_slot)], coord, result);\n"
      << "}\n";

    return s.str();
}


CompileGraphResult GraphCompiler::compile(const GraphIR& ir,
                                          const NodeLibrary& lib,
                                          VkFormat output_format) {
    CompileGraphResult result;
    if (ir.nodes.empty()) { result.error = "Graph has no nodes"; return result; }

    // 1. Assign SSBO base slots per node.
    std::unordered_map<NodeId, int> param_base_slot;
    int next_slot = 0;
    for (NodeId id : ir.eval_order) {
        auto* inst = ir.find(id);
        auto* type = lib.find(inst->type_id);
        param_base_slot[id] = next_slot;
        next_slot += (int)type->params.size();
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
        pass.output_resource = {id, 0};
        pass.param_base_slot = param_base_slot[id];
        pass.input_mode      = InputMode::PreSampled;
    
        // Classify: zero inputs AND zero params == pure resource (image source).
        // For now we treat all generators as Dispatch; only future image/external
        // nodes will set kind=ResourceBind. We expose the hook here so Phase 6
        // can flip it without touching the executor.
        const bool is_resource_node =
            (type->inputs.empty() && type->params.empty() && type->glsl_function.empty());
        pass.kind = is_resource_node ? PassKind::ResourceBind : PassKind::Dispatch;
    
        const uint32_t inputs_n = (uint32_t)type->inputs.size();
        uint32_t param_socket_count = 0;
        for (auto& p : type->params) if (p.as_socket) ++param_socket_count;
        const uint32_t total_slots = inputs_n + param_socket_count;
    
        pass.input_resources.assign(total_slots, ResourceUUID{});
        for (auto& c : ir.connections) {
            if (c.dst_node != id) continue;
            if (c.dst_socket < total_slots)
                pass.input_resources[c.dst_socket] = {c.src_node, 0};
        }
    
        if (pass.kind == PassKind::Dispatch) {
            // Build variant key BEFORE emitting GLSL — cache lookup uses this.
            pass.variant_key = build_variant_key(*type, total_slots, param_socket_count, output_format);
            pass.shader_glsl = emit_node_shader(*inst, *type, pass.variant_key, pass.param_base_slot, total_slots);
        }
        pass.input_socket_count = total_slots;
        plan.passes.push_back(std::move(pass));
    }
    plan.final_output_resource = {ir.output_node, 0};

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
        if (!seen.insert(p.output_resource).second) {
            result.success = false;
            result.error = "internal: duplicate output_resource for node "
                         + std::to_string(p.output_resource.node_id);
            return result;
        }
    }

    result.success            = true;
    result.pass_plan          = std::move(plan);
    result.param_base_slot    = std::move(param_base_slot);
    result.total_param_floats = next_slot;
    return result;
}

} // namespace te