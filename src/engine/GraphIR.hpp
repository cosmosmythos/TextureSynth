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
    // SD-style depth inheritance: how this node's BitDepth is resolved.
    DepthMode depth_mode    = DepthMode::Auto;
    BitDepth  absolute_depth = BitDepth::F16;
    // Resolved at validate_graph time (stamped for downstream stages).
    // For Auto: graph_default_depth; MatchInput: upstream's resolved depth;
    // Absolute: absolute_depth.
    BitDepth  resolved_depth = BitDepth::F16;
    // Phase 1c: muted nodes are absent after validation rewires; bypassed nodes remain so compiler can emit clear pass.
    bool muted    = false;
    bool bypassed = false;
    // Stage 2: mirror of NodeType::pass_kind, copied by validate_graph. Source of truth is NodeType (type-level, not per-instance).
    PassKind pass_kind = PassKind::Compute;
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

    // Graph-level default depth (from sidebar "Precision"). Inherited by
    // nodes whose depth_mode == Auto. SD-style graph default.
    BitDepth graph_default_depth = BitDepth::F16;

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

// Resolve a node's StorageFormat from its validated state.
// Single source of truth for node output format — used by ResourceManager,
// EnginePassCompile (intermediate allocation), and any future format consumers.
// Channels: from format_override, else from the node type's output socket
//           declaration, else RGBA.  Depth: from vn.resolved_depth.
StorageFormat resolve_node_storage(const ValidatedNode& vn,
                                   const NodeLibrary& lib,
                                   uint32_t output_index = 0);

// Validate a raw Graph against a NodeLibrary and produce a GraphIR.
GraphIRResult validate_graph(const Graph& graph, const NodeLibrary& lib);

// Resolve per-node BitDepth via SD-style inheritance. Reads ir.graph_default_depth
// (stamped by the engine after validate_graph) and each node's depth_mode.
// Must run after validate_graph and after ir.graph_default_depth is set.
void resolve_node_depths(GraphIR& ir);

} // namespace te
