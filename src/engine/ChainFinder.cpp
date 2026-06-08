#include "engine/ChainFinder.hpp"
#include "engine/Graph.hpp"        // NodeType, PassKind
#include "engine/Logging.hpp"       // log_warn
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace te {

// ---------- internal helpers ------------------------------------------------

namespace {

// Edge index for O(1) predecessor/successor lookup. Built once per
// find_chains call.
struct EdgeIndex {
    std::unordered_map<NodeId, std::vector<NodeId>> preds;
    std::unordered_map<NodeId, std::vector<NodeId>> succs;
    std::unordered_map<NodeId, uint32_t> in_degree;       // socket-0 connections (chain input)
    std::unordered_map<NodeId, uint32_t> total_in_degree;  // all connections (fan-in barrier)
    std::unordered_map<NodeId, uint32_t> out_degree;
};

EdgeIndex build_edge_index(const GraphIR& ir) {
    EdgeIndex idx;
    // Defensive: only count edges whose endpoints are in the active
    // subgraph (eval_order). The validator's BFS already prunes these,
    // but we re-check so a future change to validate_graph can't
    // silently break chain counts.
    std::unordered_set<NodeId> active;
    active.reserve(ir.eval_order.size());
    for (NodeId id : ir.eval_order) {
        active.insert(id);
        idx.preds[id]          = {};
        idx.succs[id]          = {};
        idx.in_degree[id]      = 0;
        idx.total_in_degree[id] = 0;
        idx.out_degree[id]     = 0;
    }
    for (const auto& c : ir.connections) {
        if (!active.count(c.src_node) || !active.count(c.dst_node)) continue;
        if (c.dst_socket == 0) {                       // input[0] is the chain input
            idx.preds[c.dst_node].push_back(c.src_node);
            ++idx.in_degree[c.dst_node];
        }
        ++idx.total_in_degree[c.dst_node];              // count all connections
        idx.succs[c.src_node].push_back(c.dst_node);
        ++idx.out_degree[c.src_node];
    }
    return idx;
}

// A node is a chain barrier if ANY of:
//   - its PassKind is not PurePixel (Boundary, Reduction, Feedback, etc.)
//   - it has more than one output (multi-output: split_rgba)
//   - it has more than one successor (fan-out: one producer, multiple consumers)
//   - it has more than one predecessor (fan-in: one consumer, multiple producers)
//
// Note: a node with out_degree=0 (which is typically the output_node)
// is NOT a barrier -- it is a valid chain endpoint. walk_forward
// naturally stops there.
//
// The output_node is also NOT automatically a barrier. If it is PurePixel
// (the common case), it is part of the chain ending at it. If it is a
// non-PurePixel PassKind (e.g. Readback), it becomes a barrier via the
// PassKind check above.
bool is_barrier(NodeId id, const GraphIR& ir, const NodeLibrary& lib,
                const EdgeIndex& idx) {
    const ValidatedNode* inst = ir.find(id);
    if (!inst) return true;                                       // unknown = safe default
    if (inst->pass_kind != PassKind::PurePixel) return true;     // non-fuseable PassKind
    const NodeType* type = lib.find(inst->type_id);
    if (!type || type->outputs.size() != 1) return true;         // multi-output
    auto od_it = idx.out_degree.find(id);
    if (od_it != idx.out_degree.end() && od_it->second > 1) return true;  // fan-out
    auto id_it2 = idx.total_in_degree.find(id);
    if (id_it2 != idx.total_in_degree.end() && id_it2->second > 1) return true;  // fan-in (any socket)
    return false;
}

// Identify chain HEADS: a node that starts a new chain. A node is a
// head when:
//   - it has no predecessor (source)
//   - it has multiple predecessors (fan-in node, multi-input mid-chain)
//   - its unique predecessor is a barrier
//
// The output_node is NOT specially a head. If the chain ending at the
// output_node is a PurePixel run, the output_node is its last node --
// the run is discovered by walking forward from a source. If the
// output_node is a barrier (e.g. Readback), step 5 of the algorithm
// emits it as a singleton.
//
// Barriers themselves are NOT emitted as heads here -- step 5 is the
// single source of barrier-singleton chains. This keeps the algorithm
// readable: heads = "where chains start", step 5 = "where barriers go".
std::vector<NodeId> find_chain_heads(const GraphIR& ir,
                                      const NodeLibrary& lib,
                                      const EdgeIndex& idx) {
    std::vector<NodeId> heads;
    for (NodeId id : ir.eval_order) {
        if (is_barrier(id, ir, lib, idx)) continue;   // step 5 owns barriers
        uint32_t in_d = idx.in_degree.at(id);
        if (in_d == 0) {                              // source
            heads.push_back(id);
            continue;
        }
        if (in_d > 1) {                               // fan-in
            heads.push_back(id);
            continue;
        }
        NodeId pred = idx.preds.at(id)[0];
        if (is_barrier(pred, ir, lib, idx)) {         // predecessor is a barrier
            heads.push_back(id);
        }
    }
    return heads;
}

// Walk forward from `head`, collecting a maximal run of PurePixel
// nodes. Stops BEFORE pushing a barrier (barriers become singleton
// chains in step 5, not chain members). Stops at max_length.
// Returns the chain and (via out param) the next node to re-process
// if the chain was cut short by max_chain_length (the caller must
// add it as a new head).
//
// Defensive: refuses to start from a barrier (returns empty) -- even
// though find_chain_heads no longer emits them, this guard keeps the
// function correct if called independently. A max_length of 0 is
// clamped to 1: a chain of zero nodes would violate the non-empty
// invariant, and the smallest meaningful chain is a single node.
std::vector<NodeId> walk_forward(NodeId head, const GraphIR& ir,
                                 const NodeLibrary& lib, const EdgeIndex& idx,
                                 std::unordered_set<NodeId>& visited,
                                 uint32_t max_length,
                                 NodeId& out_next_head) {
    out_next_head = 0;
    if (max_length == 0) max_length = 1;
    if (visited.count(head)) return {};                         // skip visited
    if (is_barrier(head, ir, lib, idx)) return {};              // step 5 owns barriers
    std::vector<NodeId> chain = {head};
    NodeId cur = head;
    while (chain.size() < max_length
           && idx.out_degree.at(cur) == 1
           && !is_barrier(cur, ir, lib, idx)) {
        NodeId next = idx.succs.at(cur)[0];
        if (visited.count(next)) break;                         // cross-chain guard
        if (is_barrier(next, ir, lib, idx)) break;              // barrier ends chain
        chain.push_back(next);
        cur = next;
    }
    // If we hit max_length, the next PurePixel after the chain tail
    // needs to be re-marked as a head (its predecessor is no longer
    // a barrier -- the chain just ended because of the cap).
    if (chain.size() == max_length
        && idx.out_degree.at(cur) == 1
        && !is_barrier(cur, ir, lib, idx)) {
        NodeId cand = idx.succs.at(cur)[0];
        if (!visited.count(cand) && !is_barrier(cand, ir, lib, idx)) {
            out_next_head = cand;
        }
    }
    return chain;
}

// Build a Chain struct from a node sequence (param SSBO bookkeeping).
Chain make_chain(const std::vector<NodeId>& nodes, const GraphIR& ir,
                 const NodeLibrary& lib) {
    Chain ch;
    ch.nodes = nodes;
    int off = 0;
    for (NodeId n : nodes) {
        const auto* inst = ir.find(n);
        const auto* type = inst ? lib.find(inst->type_id) : nullptr;
        ch.param_offsets.push_back(static_cast<uint32_t>(off));
        off += type ? static_cast<int>(type->params.size()) : 0;
    }
    if (!ch.param_offsets.empty()) {
        ch.param_base_slot = static_cast<int>(ch.param_offsets.front());
    }
    ch.total_params  = static_cast<uint32_t>(off);
    ch.total_inputs  = static_cast<uint32_t>(nodes.size());
    ch.total_outputs = 1u;                                      // Phase 1 invariant
    for (NodeId n : nodes) {
        const auto* inst = ir.find(n);
        if (inst && inst->bypassed) { ch.bypassed = true; break; }
    }
    return ch;
}

// Build a singleton chain for a barrier node.
Chain make_singleton(NodeId id, const GraphIR& ir, const NodeLibrary& lib) {
    Chain ch;
    ch.nodes          = {id};
    ch.param_offsets  = {0};
    ch.param_base_slot = 0;
    const auto* inst  = ir.find(id);
    const auto* type  = inst ? lib.find(inst->type_id) : nullptr;
    ch.total_params   = type ? static_cast<uint32_t>(type->params.size()) : 0u;
    ch.total_inputs   = 0;
    ch.total_outputs  = type ? static_cast<uint32_t>(type->outputs.size()) : 1u;
    if (inst) ch.bypassed = inst->bypassed;
    return ch;
}

} // anonymous namespace

// ---------- public API ------------------------------------------------------

std::vector<Chain> find_chains(const GraphIR& ir,
                               const NodeLibrary& lib,
                               const ChainFinderOptions& opts) {
    std::vector<Chain> result;
    if (ir.eval_order.empty()) return result;             // empty graph -> 0 chains

    EdgeIndex idx = build_edge_index(ir);
    const uint32_t max_length = opts.max_chain_length == 0
        ? 1u : opts.max_chain_length;
    std::vector<NodeId> heads = find_chain_heads(ir, lib, idx);
    std::unordered_set<NodeId> visited;

    // Step 4: walk forward from each head. Use an index-based loop so
    // we can append synthetic heads (for chains cut short by the
    // max_chain_length cap). walk_forward() refuses barrier heads
    // defensively, so the visited check is the only one needed here.
    for (size_t h = 0; h < heads.size(); ++h) {
        NodeId head = heads[h];
        NodeId next_head = 0;
        std::vector<NodeId> nodes = walk_forward(head, ir, lib, idx, visited,
                                                  max_length, next_head);
        if (nodes.empty()) continue;
        for (NodeId n : nodes) visited.insert(n);
        result.push_back(make_chain(nodes, ir, lib));
        if (next_head != 0 && !visited.count(next_head)) {
            heads.push_back(next_head);
        }
    }

    // Step 5: singletons for any barrier not yet in a chain.
    for (NodeId id : ir.eval_order) {
        if (is_barrier(id, ir, lib, idx) && !visited.count(id)) {
            result.push_back(make_singleton(id, ir, lib));
            visited.insert(id);
        }
    }

    // Defensive: any unvisited node (shouldn't happen in a validated graph)
    // gets a singleton chain. This guarantees the vertex-disjoint cover
    // invariant: every node appears in exactly one chain.
    for (NodeId id : ir.eval_order) {
        if (!visited.count(id)) {
            if (opts.log_cycles) {
                log_warn("ChainFinder: unvisited node " + std::to_string(id)
                         + " -- emitting singleton fallback");
            }
            result.push_back(make_singleton(id, ir, lib));
            visited.insert(id);
        }
    }

    // Deterministic ordering: stable_sort chains by the topo position
    // of their first node. The header doc promises "chains in
    // topological order of their first node" -- this enforces it.
    std::unordered_map<NodeId, size_t> topo_pos;
    topo_pos.reserve(ir.eval_order.size());
    for (size_t i = 0; i < ir.eval_order.size(); ++i) {
        topo_pos[ir.eval_order[i]] = i;
    }
    std::stable_sort(result.begin(), result.end(),
        [&topo_pos](const Chain& a, const Chain& b) {
            const size_t ai = a.nodes.empty() ? SIZE_MAX
                                              : topo_pos.at(a.nodes.front());
            const size_t bi = b.nodes.empty() ? SIZE_MAX
                                              : topo_pos.at(b.nodes.front());
            return ai < bi;
        });
    return result;
}

} // namespace te
