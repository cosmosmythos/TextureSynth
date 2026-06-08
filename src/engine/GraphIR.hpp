#pragma once
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace te {

// GraphIR -- validated, immutable internal graph representation. Types exist, sockets in bounds, output reachable, DAG, active subgraph only. Cheap to copy.

using GraphRevisionId = uint64_t;

struct ValidatedNode {
    NodeId      id       = 0;
    std::string type_id;          // references NodeType::id
    std::string debug_name;       // human-readable label for logging/debugging
    ChannelFormat format_override = ChannelFormat::RGBA;
    // Phase 1c: muted nodes are absent after validation rewires; bypassed nodes remain so compiler can emit clear pass.
    bool muted    = false;
    bool bypassed = false;
    // Stage 2: mirror of NodeType::pass_kind, copied by validate_graph. Source of truth is NodeType (type-level, not per-instance).
    PassKind pass_kind = PassKind::PurePixel;
};

struct ValidatedConnection {
    NodeId   src_node   = 0;
    uint32_t src_socket = 0;
    NodeId   dst_node   = 0;
    uint32_t dst_socket = 0;
};

struct GraphIR {
    // Nodes in topological order (sources first, output last).
    std::vector<ValidatedNode>       nodes;
    std::vector<ValidatedConnection> connections;
    NodeId                           output_node = 0;
    uint32_t                         output_socket = 0;

    // Topological evaluation order (same IDs as nodes[], but just the IDs).
    std::vector<NodeId>              eval_order;

    // Quick lookup: node_id -> index in nodes[].
    std::unordered_map<NodeId, size_t> node_index;

    // Helper: find a validated node by ID.  Returns nullptr if not found.
    const ValidatedNode* find(NodeId id) const {
        auto it = node_index.find(id);
        if (it == node_index.end()) return nullptr;
        return &nodes[it->second];
    }
};

struct GraphIRResult {
    bool        success = false;
    GraphIR     ir;
    std::string error;           // human-readable on failure
};

// Validate a raw Graph against a NodeLibrary and produce a GraphIR.
GraphIRResult validate_graph(const Graph& graph, const NodeLibrary& lib);

} // namespace te
