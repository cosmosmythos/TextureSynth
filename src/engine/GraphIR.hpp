#pragma once
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace te {

// ---------------------------------------------------------------------------
// GraphIR — Validated, immutable internal graph representation.
//
// GraphIR is the gate between "what Python/Blender gave us" and "what the
// compiler/scheduler is allowed to consume."  Every field has been checked:
//   - All node types exist in the NodeLibrary.
//   - All socket indexes are in bounds.
//   - The output node exists and is reachable.
//   - The graph is acyclic (DAG).
//   - Only the active subgraph (reachable from output) is included.
//
// GraphIR is cheap to copy (vectors of small structs) and is intended to be
// stored as a const snapshot alongside a GraphRevisionId.
// ---------------------------------------------------------------------------

using GraphRevisionId = uint64_t;

struct ValidatedNode {
    NodeId      id       = 0;
    std::string type_id;          // references NodeType::id
    std::string debug_name;       // human-readable label for logging/debugging
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

// ---------------------------------------------------------------------------
// Validate a raw Graph against a NodeLibrary and produce a GraphIR.
//
// On success:  result.success == true,  result.ir is ready for compilation.
// On failure:  result.success == false, result.error describes the problem.
// ---------------------------------------------------------------------------
GraphIRResult validate_graph(const Graph& graph, const NodeLibrary& lib);

} // namespace te
