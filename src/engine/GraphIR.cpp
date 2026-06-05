#include "engine/GraphIR.hpp"
#include <algorithm>
#include <queue>
#include <sstream>
#include <map>

namespace te {

// Internal helpers

static bool has_node(const Graph& g, NodeId id) {
    for (auto& n : g.nodes)
        if (n.id == id) return true;
    return false;
}

static bool is_muted(const std::vector<NodeInstance>& nodes, NodeId id) {
    for (auto& n : nodes)
        if (n.id == id) return n.muted;
    return false;
}

// Resolve the effective source of a muted node M's input[0]. If M has no input, returns {0,0} (severed). If source is non-muted, returns that connection. If source is also muted, recurse. Bounded by node count.
static std::pair<NodeId, uint32_t> resolve_muted_source(
    NodeId M,
    const std::vector<NodeInstance>& nodes,
    const std::vector<Connection>& conns)
{
    NodeId cur = M;
    for (size_t step = 0; step < nodes.size() + 1; ++step) {
        bool found = false;
        for (auto& c : conns) {
            if (c.dst_node == cur && c.dst_socket == 0) {
                if (!is_muted(nodes, c.src_node)) {
                    return {c.src_node, c.src_socket};
                }
                cur = c.src_node;
                found = true;
                break;
            }
        }
        if (!found) return {0, 0};  // severed
    }
    return {0, 0};  // chain too deep — treat as severed
}

// Topological sort via Kahn's algorithm. Returns node IDs in evaluation order. On cycle returns false.
static bool topo_sort_ir(
    const std::vector<ValidatedNode>& nodes,
    const std::vector<ValidatedConnection>& conns,
    std::vector<NodeId>& order)
{
    std::map<NodeId, int> in_degree;
    std::map<NodeId, std::vector<NodeId>> adj;

    for (auto& n : nodes)
        in_degree[n.id] = 0;

    for (auto& c : conns) {
        adj[c.src_node].push_back(c.dst_node);
        in_degree[c.dst_node]++;
    }

    // Use a min-heap for deterministic ordering regardless of insertion order.
    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<NodeId>> q;
    for (auto& [id, deg] : in_degree) {
        if (deg == 0) q.push(id);
    }

    while (!q.empty()) {
        NodeId cur = q.top(); q.pop();
        order.push_back(cur);
        for (NodeId next : adj[cur]) {
            if (--in_degree[next] == 0) q.push(next);
        }
    }

    return order.size() == nodes.size();
}

// validate_graph

GraphIRResult validate_graph(const Graph& graph, const NodeLibrary& lib) {
    GraphIRResult result;

    // ── 1. Empty graph ────────────────────────────────────────────
    if (graph.nodes.empty()) {
        result.error = "Graph has no nodes";
        return result;
    }

    // ── 2. Duplicate node IDs ─────────────────────────────────────
    {
        std::unordered_set<NodeId> seen;
        for (auto& n : graph.nodes) {
            if (!seen.insert(n.id).second) {
                result.error = "Duplicate node ID: " + std::to_string(n.id);
                return result;
            }
        }
    }

    // ── 3. All node types exist in the library ────────────────────
    for (auto& n : graph.nodes) {
        if (!lib.find(n.type_id)) {
            result.error = "Unknown node type '" + n.type_id +
                           "' on node " + std::to_string(n.id);
            return result;
        }
    }

    // ── 4. Output node exists ─────────────────────────────────────
    if (!has_node(graph, graph.output_node)) {
        result.error = "Output node " + std::to_string(graph.output_node) +
                       " does not exist in graph";
        return result;
    }

    // ── 5. Validate all connection endpoints ──────────────────────
    for (size_t ci = 0; ci < graph.connections.size(); ++ci) {
        auto& c = graph.connections[ci];
        // Source node exists
        if (!has_node(graph, c.src_node)) {
            result.error = "Connection " + std::to_string(ci) +
                           ": source node " + std::to_string(c.src_node) +
                           " not found";
            return result;
        }
        // Destination node exists
        if (!has_node(graph, c.dst_node)) {
            result.error = "Connection " + std::to_string(ci) +
                           ": destination node " + std::to_string(c.dst_node) +
                           " not found";
            return result;
        }
        // Source socket in bounds
        const NodeType* src_type = nullptr;
        for (auto& n : graph.nodes) {
            if (n.id == c.src_node) { src_type = lib.find(n.type_id); break; }
        }
        if (src_type && c.src_socket >= src_type->outputs.size()) {
            result.error = "Connection " + std::to_string(ci) +
                           ": source socket " + std::to_string(c.src_socket) +
                           " out of range (node type '" + src_type->id +
                           "' has " + std::to_string(src_type->outputs.size()) +
                           " outputs)";
            return result;
        }
        // Destination socket in bounds (inputs + as_socket params)
        const NodeType* dst_type = nullptr;
        for (auto& n : graph.nodes) {
            if (n.id == c.dst_node) { dst_type = lib.find(n.type_id); break; }
        }
        if (dst_type) {
            uint32_t total_inputs = static_cast<uint32_t>(dst_type->inputs.size());
            for (auto& p : dst_type->params)
                if (p.as_socket) total_inputs++;
            if (c.dst_socket >= total_inputs) {
                result.error = "Connection " + std::to_string(ci) +
                               ": destination socket " + std::to_string(c.dst_socket) +
                               " out of range (node type '" + dst_type->id +
                               "' has " + std::to_string(total_inputs) +
                               " input slots)";
                return result;
            }
        }
    }

    // ── 6. Build active subgraph (reachable from output) ──────────
    // BFS backwards from output_node through connections.
    std::unordered_set<NodeId> reachable;
    {
        std::queue<NodeId> bfs;
        bfs.push(graph.output_node);
        reachable.insert(graph.output_node);
        while (!bfs.empty()) {
            NodeId cur = bfs.front(); bfs.pop();
            for (auto& c : graph.connections) {
                if (c.dst_node == cur && reachable.find(c.src_node) == reachable.end()) {
                    reachable.insert(c.src_node);
                    bfs.push(c.src_node);
                }
            }
        }
    }

    // Rewire connections around muted nodes (Phase 1c). Redirect outgoing connections to effective input[0] source. If severed, connection dropped. Muted nodes excluded from validated node list. Bypassed nodes stay in IR.
    std::vector<Connection> rewired_conns = graph.connections;
    for (auto& n : graph.nodes) {
        if (!n.muted) continue;
        if (!reachable.count(n.id)) continue;  // not active — already pruned
        auto eff = resolve_muted_source(n.id, graph.nodes, graph.connections);
        for (auto& c : rewired_conns) {
            if (c.src_node != n.id) continue;
            if (eff.first == 0) {
                // Severed: mark for removal (filtered out in step 7).
                c.src_node = 0;
            } else {
                c.src_node   = eff.first;
                c.src_socket = eff.second;
            }
        }
    }

    // ── 7. Populate validated nodes (active subgraph, not muted) ──
    GraphIR& ir = result.ir;
    for (auto& n : graph.nodes) {
        if (!reachable.count(n.id)) continue;  // not reachable
        if (n.muted) continue;                  // rewired out
        ValidatedNode vn;
        vn.id       = n.id;
        vn.type_id  = n.type_id;
        vn.format_override = n.format_override;
        // Prefer user-supplied debug_name (Phase 1d); fall back to "type_id_id"
        vn.debug_name = n.debug_name.empty()
                      ? (n.type_id + "_" + std::to_string(n.id))
                      : n.debug_name;
        // Muted is always false here; bypassed mirrors the user's flag.
        vn.muted    = false;
        vn.bypassed = n.bypassed;
        // Stage 2: pass_kind is type-level, looked up from NodeLibrary (authoritative), not mirrored on NodeInstance.
        if (auto* nt = lib.find(n.type_id)) {
            vn.pass_kind = nt->pass_kind;
        }
        ir.nodes.push_back(vn);
    }

    // Build node_index up front as authoritative validated-node-IDs set when filtering connections. Both endpoints must be in ir.nodes -- excludes edges to muted/rewired-out nodes.
    for (size_t i = 0; i < ir.nodes.size(); ++i) {
        ir.node_index[ir.nodes[i].id] = i;
    }

    // Populate validated connections (both endpoints in ir.nodes, and
    // the source isn't the severed sentinel from rewire).
    for (auto& c : rewired_conns) {
        if (c.src_node == 0) continue;                                // severed
        if (!ir.node_index.count(c.src_node)) continue;               // not in IR
        if (!ir.node_index.count(c.dst_node)) continue;               // not in IR
        ir.connections.push_back({c.src_node, c.src_socket,
                                  c.dst_node, c.dst_socket});
    }

    ir.output_node = graph.output_node;

    // ── 8. Topological sort (cycle detection) ─────────────────────
    if (!topo_sort_ir(ir.nodes, ir.connections, ir.eval_order)) {
        result.error = "Active subgraph contains a cycle";
        return result;
    }

    // node_index was built in step 7 to serve as authoritative validated-node-IDs set for filtering ir.connections

    result.success = true;
    return result;
}

} // namespace te
