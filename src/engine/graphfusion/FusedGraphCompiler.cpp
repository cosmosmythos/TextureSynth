#include "engine/graphfusion/FusedGraphCompiler.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/FusedGraphEmitter.hpp"
#include "engine/graphfusion/FusionPlanner.hpp"
#include "engine/graphfusion/DAG.hpp"
#include "engine/ShaderVariantKey.hpp"
#include "engine/Logging.hpp"
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
    // 1) Sampler2D: source must be in a different chain (materialized as VRAM image).
    // 2) Multi-pass nodes (pass_count > 1): must be singleton chains (need intermediates).
    // Force split AFTER the predecessor so the multi-pass node becomes a chain head.
    std::vector<size_t> split_after;
    {
        std::unordered_set<NodeId> pset(path.nodes.begin(), path.nodes.end());
        for (size_t di = 0; di < path.nodes.size(); ++di) {
            NodeId dst = path.nodes[di];
            const auto* dn = ir.find(dst);
            const auto* dt = dn ? lib.find(dn->type_id) : nullptr;
            if (!dt) continue;

            // Multi-pass nodes must be singleton chains.
            if (dt->pass_count > 1 && di > 0) {
                split_after.push_back(di - 1);
            }

            for (uint32_t s = 0; s < dt->inputs.size(); ++s) {
                if (dt->inputs[s].type != SocketType::Sampler2D) continue;
                for (const auto& c : ir.connections) {
                    if (c.dst_node == dst && c.dst_socket == s && pset.count(c.src_node)) {
                        auto sit = std::find(path.nodes.begin(), path.nodes.end(), c.src_node);
                        if (sit != path.nodes.end()) {
                            size_t si = static_cast<size_t>(sit - path.nodes.begin());
                            split_after.push_back(si);
                        }
                        break;
                    }
                }
            }
        }
        std::sort(split_after.begin(), split_after.end());
        split_after.erase(std::unique(split_after.begin(), split_after.end()),
                          split_after.end());
    }

    // 4. Build DAG from active path and plan fusion groups.
    // Use REAL graph edges from ir.connections, not synthetic linear
    // adjacency. The consumer-constraint check in split_path relies on
    // dag.successors() returning actual fan-out edges to detect when an
    // intermediate node has a downstream consumer outside its group.
    // Linear-adjacency edges hide fan-out and let invalid groups form.
    std::unordered_set<NodeId> path_set(path.nodes.begin(), path.nodes.end());
    dag::DAG<NodeId>::NodeList dag_nodes(path.nodes.begin(), path.nodes.end());
    dag::DAG<NodeId>::EdgeList dag_edges;
    for (const auto& c : ir.connections) {
        if (path_set.count(c.src_node) && path_set.count(c.dst_node)) {
            dag_edges.push_back({c.src_node, c.dst_node});
        }
    }
    dag::DAG<NodeId> dag(std::move(dag_nodes), std::move(dag_edges));

    // Compute per-node register costs.
    std::vector<uint32_t> per_node_costs;
    per_node_costs.reserve(path.nodes.size());
    for (NodeId nid : path.nodes) {
        const auto* vn = ir.find(nid);
        const auto* type = vn ? lib.find(vn->type_id) : nullptr;
        if (!type) { per_node_costs.push_back(5); continue; }
        auto cost = reg::RegisterAllocator::estimate_cost(type->id);
        per_node_costs.push_back(cost.total());
    }

    // If Sampler2D boundaries exist, split the active path into sub-paths
    // and plan each independently. This ensures the source node becomes a
    // chain tail (its output is materialized as a VRAM image) so the
    // consumer chain can sample it as a TSTexture.
    auto plan_chain_group = [&](ActivePath& sub_path,
                                std::vector<uint32_t>& sub_costs) {
        std::unordered_set<NodeId> sub_set(sub_path.nodes.begin(), sub_path.nodes.end());
        dag::DAG<NodeId>::NodeList sub_dag_nodes(sub_path.nodes.begin(), sub_path.nodes.end());
        dag::DAG<NodeId>::EdgeList sub_dag_edges;
        for (const auto& c : ir.connections) {
            if (sub_set.count(c.src_node) && sub_set.count(c.dst_node))
                sub_dag_edges.push_back({c.src_node, c.dst_node});
        }
        dag::DAG<NodeId> sub_dag(std::move(sub_dag_nodes), std::move(sub_dag_edges));

        fusion::FusionPlanner planner(DEFAULT_REG_BUDGET);
        return planner.plan(sub_dag, sub_path.nodes, sub_costs);
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
            std::vector<uint32_t> sub_costs(per_node_costs.begin() + lo,
                                            per_node_costs.begin() + hi);
            auto sub_plan = plan_chain_group(sub, sub_costs);
            for (auto& g : sub_plan.groups)
                fusion_plan.groups.push_back(std::move(g));
        }
    } else {
        fusion_plan = plan_chain_group(path, per_node_costs);
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
        bool is_multipass = (group.nodes.size() == 1);
        const NodeType* mp_type = nullptr;
        if (is_multipass) {
            const auto* mp_inst = ir.find(group.nodes[0]);
            mp_type = mp_inst ? lib.find(mp_inst->type_id) : nullptr;
            if (!mp_type || mp_type->pass_count <= 1) is_multipass = false;
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

        // Phase 1c: chain is bypassed if any member node is bypassed.
        chain.bypassed = false;
        for (NodeId nid : group.nodes) {
            if (auto* vn = ir.find(nid)) {
                if (vn->bypassed) { chain.bypassed = true; break; }
            }
        }

        if (is_multipass) {
            // Multi-pass singleton chain: emit per-sub-pass GLSL.
            chain.sub_pass_count = mp_type->pass_count;
            chain.intermediate_count = mp_type->intermediate_count;
            chain.sub_pass_glsl.resize(mp_type->pass_count);
            chain.sub_pass_variant_keys.resize(mp_type->pass_count);

            const auto* inst = ir.find(group.nodes[0]);
            const uint32_t inputs_n = (uint32_t)mp_type->inputs.size();
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

            auto fused = emit_fused_subgraph(group_path, ir, lib, chain.param_base_slot,
                                             param_base_slot);
            if (fused.ok()) {
                chain.glsl = std::move(fused.source);
                chain.external_socket_masks = std::move(fused.external_socket_masks);
                chain.internal_producer_indices = std::move(fused.internal_producer_indices);
                chain.variant_key = build_fused_key(chain, ir, lib);
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

    // 7. Compute lifetimes and color classes.
    {
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

    // 8. Validate.
    if (next_slot > static_cast<int>(MAX_NODE_PARAMS)) {
        result.success = false;
        result.error = "Param budget exceeded: graph needs "
                     + std::to_string(next_slot) + " floats, max is "
                     + std::to_string(MAX_NODE_PARAMS);
        return result;
    }

    // Compute active_resources: the set of node outputs that need a VRAM image.
    // Single source of truth — the consumer graph. A resource needs an image
    // iff at least one of:
    //   (a) it's the final preview output (record_final_blit_ reads it);
    //   (b) its producer is a solo pass (chain_index_of_pass == UINT32_MAX) —
    //       dispatched as its own compute pass, must write to an image;
    //   (c) at least one consumer reads it from a DIFFERENT chain — cross-chain
    //       consumers texelFetch, and shader registers are invisible across
    //       dispatches. In-chain consumers read registers and do NOT force
    //       an image.
    // The old "chain tail gets an image" heuristic is a special case of (c)
    // (a tail's output always feeds a downstream chain or the final output)
    // and is subsumed by this scan.
    {
        std::unordered_map<NodeId, uint32_t> node_to_chain;
        node_to_chain.reserve(plan.passes.size());
        for (uint32_t ci = 0; ci < (uint32_t)plan.chains.size(); ++ci)
            for (NodeId n : plan.chains[ci].nodes)
                node_to_chain[n] = ci;

        // (a)
        plan.active_resources.insert(plan.final_output_resource);

        for (const auto& pass : plan.passes) {
            uint32_t producer_chain = node_to_chain.count(pass.node_id)
                ? node_to_chain[pass.node_id] : UINT32_MAX;
            // (b) solo passes always materialize their output.
            if (producer_chain == UINT32_MAX) {
                for (const auto& rid : pass.output_resources)
                    plan.active_resources.insert(rid);
                continue;
            }
            // (c) chain member: only materialize if some consumer is
            //     outside this chain.
            for (const auto& rid : pass.output_resources) {
                for (const auto& other : plan.passes) {
                    if (other.node_id == pass.node_id) continue;
                    uint32_t consumer_chain = node_to_chain.count(other.node_id)
                        ? node_to_chain[other.node_id] : UINT32_MAX;
                    if (consumer_chain == producer_chain) continue;
                    bool consumed = false;
                    for (const auto& in : other.input_resources) {
                        if (in == rid) { consumed = true; break; }
                    }
                    if (consumed) {
                        plan.active_resources.insert(rid);
                        break;
                    }
                }
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
