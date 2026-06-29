#pragma once
#include <vulkan/vulkan.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <array>


namespace te {


using NodeId = uint64_t;


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
        // splitmix64 avalanche over node_id + output_index.
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
enum class SocketType { Float, Vec4, Sampler2D };


// Channel axis of the format system. Orthogonal to BitDepth.
enum class ChannelFormat : uint8_t { Mono, UV, RGB, RGBA };


// Precision axis of the format system.
enum class BitDepth : uint8_t { F8, F16, F32 };


// How a node's bit depth is chosen — inheritance mode.
enum class DepthMode : uint8_t {
    Auto,        // graph default depth (from sidebar)
    MatchInput,  // upstream node's resolved depth
    Absolute,    // node.absolute_depth
};


// Full storage format descriptor. Channels × depth.
struct StorageFormat {
    ChannelFormat channels = ChannelFormat::RGBA;
    BitDepth      depth    = BitDepth::F16;
    bool operator==(const StorageFormat&) const = default;
};


struct StorageFormatInfo {
    StorageFormat storage;
    VkFormat      vk_format;
    const char*   vk_name;
    const char*   glsl_qualifier;
    uint32_t      bytes_per_pixel;
    uint8_t       logical_channels;
    uint8_t       stored_channels;
};


[[nodiscard]] const StorageFormatInfo& storage_format_info(StorageFormat fmt);
[[nodiscard]] const StorageFormatInfo* storage_format_info_table(size_t& count);
[[nodiscard]] VkFormat storage_format_to_vk(StorageFormat fmt);
[[nodiscard]] uint32_t storage_format_bytes(StorageFormat fmt);
[[nodiscard]] std::string storage_format_glsl_qualifier(StorageFormat fmt);
[[nodiscard]] const char* vk_format_name(VkFormat fmt);
[[nodiscard]] uint32_t vk_format_bytes(VkFormat fmt);


// Socket definition.
struct Socket {
    std::string name;
    float default_value = 0.0f;
    std::array<float, 4> default_vec4 = {0.0f, 0.0f, 0.0f, 0.0f};
    SocketType type = SocketType::Vec4;
    ChannelFormat format = ChannelFormat::RGBA;
};


// Compute/Upload/Readback. Lives here so NodeType can reference it without an include cycle.
enum class PassKind : uint8_t {
    Compute,  // has a compute shader — dispatched normally
    Upload,   // CPU → GPU source — no shader, no dispatch
    Readback, // GPU → CPU terminal — no shader, no dispatch
};


struct NodeParam {
    std::string name;
    std::string display_name;
    std::string description;
    float default_value  = 0.0f;
    float min_value      = 0.0f;
    float max_value      = 1.0f;
    float soft_min_value = 0.0f;
    float soft_max_value = 1.0f;
    float step           = 0.0f;
    bool  is_integer     = false;
    bool  as_socket      = false;
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

    // How this node participates in chain fusion (set by NodeRegistryLoader).
    PassKind pass_kind = PassKind::Compute;

    // Multi-pass: pass_count>1 allocates intermediate images between sub-passes.
    uint32_t pass_count        = 1;
    uint32_t intermediate_count = 0;
};


struct NodeInstance {
    NodeId id = 0;
    std::string type_id;
    ChannelFormat format_override = ChannelFormat::RGBA;
    std::string debug_name;
    bool muted    = false;
    bool bypassed = false;
    // SD-style depth inheritance. Appended last to preserve aggregate-init
    // ordering for test/code sites using {id, type, fmt, name, m, b}.
    DepthMode depth_mode   = DepthMode::Auto;
    BitDepth  absolute_depth = BitDepth::F16;
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
    // The "active" node whose output[output_socket] is shown in the preview.
    // Clicking a node re-submits the graph with that node as active.
    NodeId output_node = 0;
    uint32_t output_socket = 0;          // which output of output_node to read for preview
    // Named "bake" targets. Each target names which output of which source node
    // to render. The Bake button iterates these and reads back each one.
    struct OutputTarget {
        NodeId source_node = 0;
        uint32_t source_socket = 0;      // output index of source_node to read
        std::string name;                // "Base Color", "Normal", "Height", ...
    };
    std::vector<OutputTarget> output_targets;
};


} // namespace te
