#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <functional>

namespace te {

using NodeId = uint64_t;

// === Phase 6: Resource identity ===
struct ResourceUUID {
    uint64_t node_id      = 0;
    uint32_t output_index = 0;

    bool operator==(const ResourceUUID& o) const noexcept {
        return node_id == o.node_id && output_index == o.output_index;
    }
    bool operator!=(const ResourceUUID& o) const noexcept { return !(*this == o); }
};

struct ResourceUUIDHash {
    size_t operator()(const ResourceUUID& r) const noexcept {
        // splitmix64: high-quality avalanche for combining two integers.
        // Future-proof for multi-output nodes (output_index > 0).
        uint64_t x = r.node_id + 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        x ^= (x >> 31);
        x ^= (uint64_t)r.output_index * 0xd1b54a32d192ed03ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x ^= (x >> 27);
        return (size_t)x;
    }
};

// Shared upper bound for the per-graph parameter SSBO. Defined here so
// GraphCompiler can bounds-check without pulling in Engine.hpp.
constexpr uint32_t MAX_NODE_PARAMS = 8192;

// --- Node type definitions (the "blueprint") ---

enum class SocketType { Float, Vec4, Sampler2D };

struct Socket {
    std::string name;   // e.g. "color", "height"
    SocketType type = SocketType::Vec4;
};

struct NodeParam {
    std::string name;
    std::string display_name;
    std::string description;
    float default_value = 0.0f;
    float min_value     = 0.0f;
    float max_value     = 1.0f;
    float step          = 0.0f;
    bool  is_integer    = false;
    bool  as_socket     = false;
};

struct NodeType {
    std::string id;
    std::string display_name;
    std::string description;
    std::string glsl_function;
    std::vector<Socket> inputs;
    std::vector<Socket> outputs;
    std::vector<NodeParam> params;

    // Feature flags this node respects (from .node.json "variant_flags").
    // Empty = no compile-time variants. Populated by NodeRegistryLoader.
    std::vector<std::string> variant_flags;
};

// --- Graph instance (the "scene") ---

struct NodeInstance {
    NodeId id = 0;
    std::string type_id;                // references NodeType::id
};

struct Connection {
    NodeId src_node = 0;
    uint32_t src_socket = 0;            // index into NodeType::outputs
    NodeId dst_node = 0;
    uint32_t dst_socket = 0;            // index into NodeType::inputs
};

struct Graph {
    std::vector<NodeInstance> nodes;
    std::vector<Connection> connections;
    NodeId output_node = 0;             // which node's output[0] goes to imageStore
};

} // namespace te
