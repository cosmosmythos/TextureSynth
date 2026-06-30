#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/FusionPlanner.hpp"
#include "engine/graphfusion/DAG.hpp"
#include "engine/ShaderVariantKey.hpp"
#include "engine/Logging.hpp"
#include "engine/register_allocation/LivenessAnalysis.hpp"
#include "engine/register_allocation/InterferenceGraph.hpp"
#include "engine/register_allocation/GraphColorer.hpp"
#include "engine/memory_allocation/AliasColorer.hpp"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace te {

namespace {

// Build a ShaderVariantKey for a single-node pass (same as GraphCompiler).
// feature_flags: bits 0..2 = ChannelFormat, bits 3..4 = BitDepth.
ShaderVariantKey build_variant_key(const NodeType& type,
                                   uint32_t input_count,
                                   uint32_t param_socket_count,
                                   ChannelFormat format,
                                   BitDepth depth,
                                   uint32_t sub_pass_index = 0,
                                   uint32_t pass_count = 1) {
    ShaderVariantKey key;
    key.node_type_id = type.id;
    key.input_count = input_count;

    uint32_t mask = 0;
    for (uint32_t i = 0; i < param_socket_count && i < 32; ++i) {
        if (i < type.params.size() && type.params[i].as_socket)
            mask |= (1u << i);
    }
    key.param_socket_mask = mask;
    const uint32_t fmt_bits   = static_cast<uint32_t>(format) & 0x7u;
    const uint32_t depth_bits = static_cast<uint32_t>(depth)  & 0x3u;
    key.feature_flags = fmt_bits | (depth_bits << 3);

    if (pass_count > 1) {
        key.specialization[0] = sub_pass_index;
        key.specialization_count = 1;
    }
    return key;
}

// Build a FusedVariantKey for a chain (fused shader cache lookup).
// Per-node packing in feature_flags: 3 bits format_override + 2 bits depth
// = 5 bits per node. A 32-bit word holds 6 nodes; longer chains wrap onto
// additional implicit bits via OR-folding (matches the original design).
FusedVariantKey build_fused_key(const Chain& chain,
                                const GraphIR& ir,
                                const NodeLibrary& lib) {
    FusedVariantKey k;
    uint32_t feature = 0;
    uint32_t shift = 0;
    for (NodeId n : chain.nodes) {
        const auto* inst = ir.find(n);
        const auto* type = inst ? lib.find(inst->type_id) : nullptr;
        if (!type) continue;
        k.node_type_ids.push_back(type->id);
        uint32_t mask = 0;
        for (uint32_t i = 0; i < type->params.size() && i < 32; ++i) {
            if (type->params[i].as_socket) mask |= (1u << i);
        }
        k.param_socket_masks.push_back(mask);
        k.input_counts.push_back(static_cast<uint32_t>(type->inputs.size()));
        if (inst) {
            const ChannelFormat resolved_fmt = resolve_node_storage(*inst, lib).channels;
            feature |= (static_cast<uint32_t>(resolved_fmt) & 0x7u) << shift;
            shift += 3;
            feature |= (static_cast<uint32_t>(inst->resolved_depth) & 0x3u) << shift;
            shift += 2;
        }
    }
    k.feature_flags = feature;
    k.external_socket_masks = chain.external_socket_masks;
    k.internal_producer_indices = chain.internal_producer_indices;
    return k;
}

} // namespace

CompileGraphResult FusedGraphCompiler::compile(const GraphIR& ir,
                                               const NodeLibrary& lib,
                                               NodeId active_node_id) {
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

    // 2. Build per-node ComputePass entries (same as GraphCompiler).
    PassPlan plan;
    plan.passes.reserve(ir.eval_order.size());

    for (NodeId id : ir.eval_order) {
        auto* inst = ir.find(id);
        auto* type = lib.find(inst->type_id);

        ComputePass pass;
        pass.node_id         = id;
        pass.type_id         = type->id;
        pass.output_resources.clear();
        for (uint32_t i = 0; i < type->outputs.size(); ++i)
            pass.output_resources.push_back({id, i});
        pass.param_base_slot = param_base_slot[id];
        pass.input_mode      = InputMode::PreSampled;
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
            pass.variant_key = build_variant_key(*type, total_slots, param_socket_count,
                                                 inst->format_override, inst->resolved_depth,
                                                 0, type->pass_count);
            pass.shader_glsl = ""; // per-node shader not emitted here; chain shader replaces it
        }
        pass.input_socket_count = total_slots;
        plan.passes.push_back(std::move(pass));
    }
    plan.final_output_resource = {ir.output_node, ir.output_socket};

    // 3. Trace active path.
    auto path = ActivePathTracer::trace(ir, active_node_id, lib);
    if (path.nodes.empty()) {
        // No active path — fall back to per-node passes with no chains.
        plan.chain_index_of_pass.assign(plan.passes.size(), UINT32_MAX);
        // Emit per-node GLSL so passes have valid shader source.
        for (uint32_t i = 0; i < (uint32_t)plan.passes.size(); ++i) {
            auto& pass = plan.passes[i];
            if (pass.kind != PassKind::Compute || pass.bypassed) continue;
            const auto* inst = ir.find(pass.node_id);
            const auto* type = lib.find(pass.type_id);
            if (!inst || !type) continue;
            pass.shader_glsl = emit_node_shader(*inst, *type, pass.variant_key,
                                                pass.param_base_slot,
                                                pass.input_socket_count,
                                                inst->format_override,
                                                pass.input_resources);
        }
        result.success = true;
        result.pass_plan = std::move(plan);
        result.param_base_slot = std::move(param_base_slot);
        result.total_param_floats = next_slot;
        return result;
    }

    // 3b. Detect boundaries that force chain splits:
    //   - Multi-pass nodes (pass_count > 1): singleton chains only.
    //   - Sampler2D inputs: consumer must be in a different chain from its source.
    std::vector<size_t> split_after;
    for (size_t di = 0; di < path.nodes.size(); ++di) {
        const auto* inst = ir.find(path.nodes[di]);
        const auto* type = inst ? lib.find(inst->type_id) : nullptr;
        if (!type) continue;

        if (type->pass_count > 1) {
            if (di > 0)                split_after.push_back(di - 1);
            if (di + 1 < path.nodes.size()) split_after.push_back(di);
        }

        bool has_sampler = false;
        for (const auto& inp : type->inputs)
            if (inp.type == SocketType::Sampler2D) { has_sampler = true; break; }
        if (has_sampler && di > 0)
            split_after.push_back(di - 1);
    }
    std::sort(split_after.begin(), split_after.end());
    split_after.erase(std::unique(split_after.begin(), split_after.end()),
                      split_after.end());

    // 4. Build DAG from active path and plan fusion groups.
    // Use REAL graph edges from ir.connections (not synthetic linear adjacency)
    // so the planner's consumer-constraint check sees true fan-out.
    auto build_dag = [&](const std::vector<NodeId>& nodes) {
        std::unordered_set<NodeId> node_set(nodes.begin(), nodes.end());
        dag::DAG<NodeId>::NodeList dnodes(nodes.begin(), nodes.end());
        dag::DAG<NodeId>::EdgeList dedges;
        for (const auto& c : ir.connections)
            if (node_set.count(c.src_node) && node_set.count(c.dst_node))
                dedges.push_back({c.src_node, c.dst_node});
        return dag::DAG<NodeId>(std::move(dnodes), std::move(dedges));
    };

    auto plan_chain_group = [&](ActivePath& sub_path) {
        auto sub_dag = build_dag(sub_path.nodes);
        auto pressure_fn = [&](const std::vector<NodeId>& range) -> uint32_t {
            std::vector<ResourceUUID> ext_outputs;
            if (!range.empty()) {
                const auto* t = ir.find(range.back());
                const auto* tt = t ? lib.find(t->type_id) : nullptr;
                if (tt)
                    for (uint32_t i = 0; i < tt->outputs.size(); ++i)
                        ext_outputs.push_back(ResourceUUID{range.back(), i});
            }
            auto intervals = register_allocation::LivenessAnalysis::compute_intervals(
                range, ir.connections, ext_outputs);
            return register_allocation::GraphColorer::color_linear_scan(
                intervals, range, DEFAULT_REG_BUDGET).colors_used;
        };
        fusion::FusionPlanner planner(DEFAULT_REG_BUDGET);
        return planner.plan(sub_dag, sub_path.nodes, pressure_fn);
    };

    fusion::FusionPlan<NodeId> fusion_plan;
    if (!split_after.empty()) {
        std::vector<size_t> bounds = {0};
        for (size_t sa : split_after) bounds.push_back(sa + 1);
        bounds.push_back(path.nodes.size());

        for (size_t b = 0; b < bounds.size() - 1; ++b) {
            size_t lo = bounds[b], hi = bounds[b + 1];
            if (lo >= hi) continue;
            ActivePath sub;
            sub.nodes.assign(path.nodes.begin() + lo, path.nodes.begin() + hi);
            auto sub_plan = plan_chain_group(sub);
            for (auto& g : sub_plan.groups)
                fusion_plan.groups.push_back(std::move(g));
        }
    } else {
        fusion_plan = plan_chain_group(path);
    }

    // 5. Emit fused GLSL for each fusion group and create Chain entries.
    // Map node_id -> pass index for chain_index_of_pass.
    std::unordered_map<NodeId, uint32_t> pass_idx_by_node;
    pass_idx_by_node.reserve(plan.passes.size());
    for (uint32_t i = 0; i < (uint32_t)plan.passes.size(); ++i)
        pass_idx_by_node[plan.passes[i].node_id] = i;

    plan.chains.reserve(fusion_plan.groups.size());
    for (auto& group : fusion_plan.groups) {
        Chain chain;
        chain.nodes = group.nodes;

        // Check if this is a singleton multi-pass node.
        const NodeType* mp_type = nullptr;
        bool is_multipass = false;
        if (group.nodes.size() == 1) {
            const auto* inst = ir.find(group.nodes[0]);
            mp_type = inst ? lib.find(inst->type_id) : nullptr;
            is_multipass = mp_type && mp_type->pass_count > 1;
        }

        // Compute param offsets and base slot.
        // chain_base = min global slot so all nodes get non-negative relative offsets.
        uint32_t min_slot = UINT32_MAX;
        for (NodeId nid : group.nodes) {
            auto it = param_base_slot.find(nid);
            if (it != param_base_slot.end() && it->second >= 0)
                min_slot = std::min(min_slot, static_cast<uint32_t>(it->second));
        }
        if (min_slot == UINT32_MAX) min_slot = 0;
        chain.param_base_slot = min_slot;
        chain.param_offsets.reserve(group.nodes.size());
        chain.param_global_slots.reserve(group.nodes.size());
        uint32_t running_offset = 0;
        for (NodeId nid : group.nodes) {
            chain.param_offsets.push_back(running_offset);
            chain.param_global_slots.push_back(
                static_cast<uint32_t>(param_base_slot[nid]));
            const auto* type = lib.find(ir.find(nid)->type_id);
            uint32_t float_input_count = 0;
            for (const auto& inp : type->inputs)
                if (inp.type == SocketType::Float) ++float_input_count;
            running_offset += (uint32_t)type->params.size() + float_input_count;
        }
        chain.total_params = running_offset;

        // Chain is bypassed if any member node is bypassed.
        chain.bypassed = false;
        for (NodeId nid : group.nodes) {
            if (auto* vn = ir.find(nid); vn && vn->bypassed) {
                chain.bypassed = true;
                break;
            }
        }

        if (is_multipass) {
            // Multi-pass singleton chain: emit per-sub-pass GLSL.
            chain.sub_pass_count = mp_type->pass_count;
            chain.intermediate_count = mp_type->intermediate_count;
            chain.sub_pass_glsl.resize(mp_type->pass_count);
            chain.sub_pass_variant_keys.resize(mp_type->pass_count);

            const auto* inst = ir.find(group.nodes[0]);
            uint32_t param_socket_count = 0;
            for (auto& p : mp_type->params) if (p.as_socket) ++param_socket_count;

            auto& pass = plan.passes[pass_idx_by_node[group.nodes[0]]];
            for (uint32_t sp = 0; sp < mp_type->pass_count; ++sp) {
                chain.sub_pass_variant_keys[sp] = build_variant_key(
                    *mp_type, pass.input_socket_count, param_socket_count,
                    inst->format_override, inst->resolved_depth,
                    sp, mp_type->pass_count);
                chain.sub_pass_glsl[sp] = emit_node_shader(
                    *inst, *mp_type, chain.sub_pass_variant_keys[sp],
                    pass.param_base_slot, pass.input_socket_count,
                    inst->format_override, pass.input_resources);
            }
            chain.glsl = pass.shader_glsl;
        } else {
            // Fused chain: emit single fused GLSL.
            ActivePath group_path;
            group_path.nodes = group.nodes;

            // Compute register allocation for this chain.
            {
                std::vector<ResourceUUID> ext_outputs;
                NodeId tail = group.nodes.back();
                const auto* tail_inst = ir.find(tail);
                const auto* tail_type = tail_inst ? lib.find(tail_inst->type_id) : nullptr;
                if (tail_type)
                    for (uint32_t i = 0; i < tail_type->outputs.size(); ++i)
                        ext_outputs.push_back(ResourceUUID{tail, i});
                auto intervals = register_allocation::LivenessAnalysis::compute_intervals(
                    group.nodes, ir.connections, ext_outputs);
                chain.coloring = register_allocation::GraphColorer::color_linear_scan(
                    intervals, group.nodes, DEFAULT_REG_BUDGET);
            }

            auto fused = emit_fused_subgraph(group_path, ir, lib, chain.param_base_slot,
                                             param_base_slot, &chain.coloring,
                                             &plan.active_resources);
            if (fused.ok()) {
                chain.glsl = std::move(fused.source);
                chain.external_socket_masks = std::move(fused.external_socket_masks);
                chain.internal_producer_indices = std::move(fused.internal_producer_indices);
                chain.variant_key = build_fused_key(chain, ir, lib);
                log_info("FusedGraphCompiler: chain [" +
                         [&]{ std::string s; for (auto n : group.nodes) s += std::to_string(n) + ","; return s; }() +
                         "] regs=" + std::to_string(chain.coloring.colors_used) +
                         " spilled=" + std::to_string(chain.coloring.spilled.size()) +
                         " shared_slots=" + std::to_string(chain.coloring.shared_slot_count));
            } else {
                log_warn("FusedGraphCompiler: group [" +
                         [&]{ std::string s; for (auto n : group.nodes) s += std::to_string(n) + ","; return s; }() +
                         "] emit failed: " + fused.error);
            }
        }

        plan.chains.push_back(std::move(chain));
    }

    // 6. Fill chain_index_of_pass.
    plan.chain_index_of_pass.assign(plan.passes.size(), UINT32_MAX);
    for (size_t ci = 0; ci < plan.chains.size(); ++ci) {
        for (NodeId n : plan.chains[ci].nodes) {
            auto it = pass_idx_by_node.find(n);
            if (it != pass_idx_by_node.end())
                plan.chain_index_of_pass[it->second] = (uint32_t)ci;
        }
    }

    // 6b. Emit per-node GLSL for passes not in any chain.
    for (uint32_t i = 0; i < (uint32_t)plan.passes.size(); ++i) {
        if (plan.chain_index_of_pass[i] != UINT32_MAX) continue;
        auto& pass = plan.passes[i];
        if (pass.kind != PassKind::Compute || pass.bypassed) continue;
        const auto* inst = ir.find(pass.node_id);
        const auto* type = lib.find(pass.type_id);
        if (!inst || !type) continue;
        pass.shader_glsl = emit_node_shader(*inst, *type, pass.variant_key,
                                            pass.param_base_slot,
                                            pass.input_socket_count,
                                            inst->format_override,
                                            pass.input_resources);
    }

    // 7. Compute lifetimes and color classes via AliasColorer.
    {
        auto alias_result = memory_allocation::AliasColorer::compute(
            plan.passes, plan.final_output_resource, ir, lib);
        plan.lifetimes.reserve(alias_result.lifetimes.size());
        for (const auto& kv : alias_result.lifetimes) {
            auto& lt = plan.lifetimes[kv.first];
            lt.first_pass = kv.second.first_pass;
            lt.last_pass = kv.second.last_pass;
        }
        plan.color_classes = std::move(alias_result.color_classes);
    }

    // 8. Validate.
    if (next_slot > static_cast<int>(MAX_NODE_PARAMS)) {
        result.success = false;
        result.error = "Param budget exceeded: graph needs "
                     + std::to_string(next_slot) + " floats, max is "
                     + std::to_string(MAX_NODE_PARAMS);
        return result;
    }

    // 9. Compute active_resources: node outputs that need a VRAM image.
    // A resource needs an image iff:
    //   (a) it's the final preview output;
    //   (b) its producer is a solo pass (not in any chain);
    //   (c) at least one consumer is in a DIFFERENT chain.
    {
        std::unordered_map<NodeId, uint32_t> node_to_chain;
        for (uint32_t ci = 0; ci < (uint32_t)plan.chains.size(); ++ci)
            for (NodeId n : plan.chains[ci].nodes)
                node_to_chain[n] = ci;

        plan.active_resources.insert(plan.final_output_resource);

        auto get_chain = [&](NodeId id) -> uint32_t {
            auto it = node_to_chain.find(id);
            return it != node_to_chain.end() ? it->second : UINT32_MAX;
        };

        for (const auto& pass : plan.passes) {
            uint32_t producer_chain = get_chain(pass.node_id);
            if (producer_chain == UINT32_MAX) {
                for (const auto& rid : pass.output_resources)
                    plan.active_resources.insert(rid);
                continue;
            }
            for (const auto& rid : pass.output_resources) {
                for (const auto& other : plan.passes) {
                    if (other.node_id == pass.node_id) continue;
                    if (get_chain(other.node_id) == producer_chain) continue;
                    for (const auto& in : other.input_resources) {
                        if (in == rid) {
                            plan.active_resources.insert(rid);
                            goto next_resource;
                        }
                    }
                }
                next_resource:;
            }
        }
    }

    result.success = true;
    result.pass_plan = std::move(plan);
    result.param_base_slot = std::move(param_base_slot);
    result.total_param_floats = next_slot;
    return result;
}

} // namespace te
