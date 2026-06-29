#include "engine/GraphIR.hpp"
#include <algorithm>
#include <queue>
#include <map>
#include <set>
#include <unordered_set>

namespace te {


StorageFormat resolve_node_storage(const ValidatedNode& vn,
                                   const NodeLibrary& lib,
                                   uint32_t output_index) {
    ChannelFormat ch = vn.format_override;
    auto* type = lib.find(vn.type_id);
    if (type && output_index < type->outputs.size()
        && type->outputs[output_index].format != ChannelFormat::RGBA) {
        ch = type->outputs[output_index].format;
    } else if (ch == ChannelFormat::RGBA) {
        if (type && !type->outputs.empty()) {
            ch = type->outputs[0].format;
        }
    }
    return StorageFormat{ch, vn.resolved_depth};
}


static const NodeType* find_node_type(const Graph& g, const NodeLibrary& lib, NodeId id) {
    for (const auto& n : g.nodes)
        if (n.id == id) return lib.find(n.type_id);
    return nullptr;
}


static bool is_muted(const std::vector<NodeInstance>& nodes, NodeId id) {
    for (const auto& n : nodes)
        if (n.id == id) return n.muted;
    return false;
}


// Sentinel: connection severed by muted-node rewire.
constexpr NodeId SEVERED = ~NodeId{0};


[[nodiscard]] static std::pair<NodeId, uint32_t> resolve_muted_source(
    NodeId M, const std::vector<NodeInstance>& nodes, const std::vector<Connection>& conns) {
    NodeId cur = M;
    for (size_t step = 0; step < nodes.size() + 1; ++step) {
        bool found = false;
        std::set<uint32_t> checked_sockets;
        for (const auto& c : conns) {
            if (c.dst_node != cur) continue;
            if (checked_sockets.count(c.dst_socket)) continue;
            checked_sockets.insert(c.dst_socket);
            if (!is_muted(nodes, c.src_node))
                return {c.src_node, c.src_socket};
            cur = c.src_node;
            found = true;
            break;
        }
        if (!found) return {SEVERED, 0};
    }
    return {SEVERED, 0};
}


// Kahn's topo sort. Returns false on cycle.
[[nodiscard]] static bool topo_sort_ir(
    const std::vector<ValidatedNode>& nodes,
    const std::vector<ValidatedConnection>& conns,
    std::vector<NodeId>& order)
{
    std::map<NodeId, int> in_degree;
    std::map<NodeId, std::vector<NodeId>> adj;

    for (const auto& n : nodes)
        in_degree[n.id] = 0;

    for (const auto& c : conns) {
        adj[c.src_node].push_back(c.dst_node);
        in_degree[c.dst_node]++;
    }

    // Min-heap for deterministic ordering regardless of insertion order.
    std::priority_queue<NodeId, std::vector<NodeId>, std::greater<NodeId>> q;
    for (const auto& [id, deg] : in_degree) {
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
        for (const auto& n : graph.nodes) {
            if (!seen.insert(n.id).second) {
                result.error = "Duplicate node ID: " + std::to_string(n.id);
                return result;
            }
        }
    }

    // ── 3. All node types exist in the library ────────────────────
    for (const auto& n : graph.nodes) {
        if (!lib.find(n.type_id)) {
            result.error = "Unknown node type '" + n.type_id +
                           "' on node " + std::to_string(n.id);
            return result;
        }
    }

    // ── 4. Output node exists ─────────────────────────────────────
    if (!find_node_type(graph, lib, graph.output_node) &&
        !std::any_of(graph.nodes.begin(), graph.nodes.end(),
                     [&](const NodeInstance& n){ return n.id == graph.output_node; })) {
        result.error = "Output node " + std::to_string(graph.output_node) +
                       " does not exist in graph";
        return result;
    }

    // ── 5. Validate all connection endpoints ──────────────────────
    for (size_t ci = 0; ci < graph.connections.size(); ++ci) {
        const auto& c = graph.connections[ci];
        if (!std::any_of(graph.nodes.begin(), graph.nodes.end(),
                         [&](const NodeInstance& n){ return n.id == c.src_node; })) {
            result.error = "Connection " + std::to_string(ci) +
                           ": source node " + std::to_string(c.src_node) + " not found";
            return result;
        }
        if (!std::any_of(graph.nodes.begin(), graph.nodes.end(),
                         [&](const NodeInstance& n){ return n.id == c.dst_node; })) {
            result.error = "Connection " + std::to_string(ci) +
                           ": destination node " + std::to_string(c.dst_node) + " not found";
            return result;
        }
        const NodeType* src_type = find_node_type(graph, lib, c.src_node);
        if (src_type && c.src_socket >= src_type->outputs.size()) {
            result.error = "Connection " + std::to_string(ci) +
                           ": source socket " + std::to_string(c.src_socket) +
                           " out of range (node type '" + src_type->id +
                           "' has " + std::to_string(src_type->outputs.size()) + " outputs)";
            return result;
        }
        const NodeType* dst_type = find_node_type(graph, lib, c.dst_node);
        if (dst_type) {
            uint32_t total_inputs = static_cast<uint32_t>(dst_type->inputs.size());
            for (const auto& p : dst_type->params)
                if (p.as_socket) total_inputs++;
            if (c.dst_socket >= total_inputs) {
                result.error = "Connection " + std::to_string(ci) +
                               ": destination socket " + std::to_string(c.dst_socket) +
                               " out of range (node type '" + dst_type->id +
                               "' has " + std::to_string(total_inputs) + " input slots)";
                return result;
            }
        }
    }

    // ── 6. Build active subgraph (reachable from output) ──────────
    std::unordered_set<NodeId> reachable;
    {
        std::queue<NodeId> bfs;
        bfs.push(graph.output_node);
        reachable.insert(graph.output_node);
        while (!bfs.empty()) {
            NodeId cur = bfs.front(); bfs.pop();
            for (const auto& c : graph.connections) {
                if (c.dst_node == cur && reachable.find(c.src_node) == reachable.end()) {
                    reachable.insert(c.src_node);
                    bfs.push(c.src_node);
                }
            }
        }
    }

    // Rewire outgoing connections from muted nodes to their effective source.
    std::vector<Connection> rewired_conns = graph.connections;
    for (const auto& n : graph.nodes) {
        if (!n.muted) continue;
        auto eff = resolve_muted_source(n.id, graph.nodes, graph.connections);
        for (auto& c : rewired_conns) {
            if (c.src_node != n.id) continue;
            if (eff.first == SEVERED) {
                c.src_node = SEVERED;
            } else {
                c.src_node   = eff.first;
                c.src_socket = eff.second;
            }
        }
    }

    // ── 7. Populate validated nodes (ALL nodes, not just reachable) ──
    // All nodes go into the IR so set_active_node can switch output
    // to any node without recompiling the graph from scratch.
    GraphIR& ir = result.ir;
    for (const auto& n : graph.nodes) {
        if (n.muted) continue;
        ValidatedNode vn;
        vn.id       = n.id;
        vn.type_id  = n.type_id;
        vn.format_override = n.format_override;
        vn.depth_mode      = n.depth_mode;
        vn.absolute_depth  = n.absolute_depth;
        vn.debug_name = n.debug_name.empty()
                      ? (n.type_id + "_" + std::to_string(n.id))
                      : n.debug_name;
        vn.muted    = false;
        vn.bypassed = n.bypassed;
        if (auto* nt = lib.find(n.type_id))
            vn.pass_kind = nt->pass_kind;
        ir.nodes.push_back(vn);
    }

    // Build node_index for connection filtering — both endpoints must exist.
    for (size_t i = 0; i < ir.nodes.size(); ++i)
        ir.node_index[ir.nodes[i].id] = i;

    for (const auto& c : rewired_conns) {
        if (c.src_node == SEVERED) continue;
        if (!ir.node_index.count(c.src_node)) continue;
        if (!ir.node_index.count(c.dst_node)) continue;
        ir.connections.push_back({c.src_node, c.src_socket,
                                  c.dst_node, c.dst_socket});
    }

    // If output_node is muted, follow rewire to effective source.
    // If severed, fall back to first IR node.
    {
        NodeId on = graph.output_node;
        while (is_muted(graph.nodes, on)) {
            auto eff = resolve_muted_source(on, graph.nodes, graph.connections);
            if (eff.first == SEVERED) break;
            on = eff.first;
        }
        if (!ir.node_index.empty() && !ir.node_index.count(on)) {
            on = ir.nodes.front().id;
        } else if (ir.node_index.empty()) {
            on = 0;
        }
        ir.output_node = on;
        ir.output_socket = graph.output_socket;
    }

    // ── 8. Topological sort (cycle detection) ─────────────────────
    if (!topo_sort_ir(ir.nodes, ir.connections, ir.eval_order)) {
        result.error = "Active subgraph contains a cycle";
        return result;
    }

    result.success = true;
    return result;
}


// Resolve BitDepth inheritance in eval_order.
// Called after graph_default_depth is set.
void resolve_node_depths(GraphIR& ir) {
    for (NodeId nid : ir.eval_order) {
        auto it = ir.node_index.find(nid);
        if (it == ir.node_index.end()) continue;
        ValidatedNode& vn = ir.nodes[it->second];
        switch (vn.depth_mode) {
            case DepthMode::Absolute:
                vn.resolved_depth = vn.absolute_depth;
                break;
            case DepthMode::MatchInput: {
                BitDepth picked = ir.graph_default_depth;
                for (const auto& c : ir.connections) {
                    if (c.dst_node != nid) continue;
                    if (auto* src = ir.find(c.src_node))
                        picked = src->resolved_depth;
                    break;
                }
                vn.resolved_depth = picked;
                break;
            }
            case DepthMode::Auto:
            default:
                vn.resolved_depth = ir.graph_default_depth;
                break;
        }
    }
}


} // namespace te
