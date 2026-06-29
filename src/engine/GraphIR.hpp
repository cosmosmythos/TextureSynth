#pragma once
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace te {

using GraphRevisionId = uint64_t;

struct ValidatedNode {
    NodeId      id       = 0;
    std::string type_id;
    std::string debug_name;
    ChannelFormat format_override = ChannelFormat::RGBA;
    DepthMode depth_mode    = DepthMode::Auto;
    BitDepth  absolute_depth = BitDepth::F16;
    BitDepth  resolved_depth = BitDepth::F16;
    bool muted    = false;
    bool bypassed = false;
    PassKind pass_kind = PassKind::Compute;
};

struct ValidatedConnection {
    NodeId   src_node   = 0;
    uint32_t src_socket = 0;
    NodeId   dst_node   = 0;
    uint32_t dst_socket = 0;
};

struct GraphIR {
    std::vector<ValidatedNode>       nodes;
    std::vector<ValidatedConnection> connections;
    NodeId                           output_node = 0;
    uint32_t                         output_socket = 0;
    BitDepth graph_default_depth = BitDepth::F16;
    std::vector<NodeId>              eval_order;
    std::unordered_map<NodeId, size_t> node_index;

    [[nodiscard]] const ValidatedNode* find(NodeId id) const {
        auto it = node_index.find(id);
        if (it == node_index.end()) return nullptr;
        return &nodes[it->second];
    }
};

struct GraphIRResult {
    bool        success = false;
    GraphIR     ir;
    std::string error;
};

// Resolve output StorageFormat for a validated node.
[[nodiscard]] StorageFormat resolve_node_storage(const ValidatedNode& vn,
                                                 const NodeLibrary& lib,
                                                 uint32_t output_index = 0);

// Validate a Graph, producing a GraphIR or error.
[[nodiscard]] GraphIRResult validate_graph(const Graph& graph, const NodeLibrary& lib);

// Resolve BitDepth inheritance. Must run after validate_graph + graph_default_depth set.
void resolve_node_depths(GraphIR& ir);

} // namespace te
