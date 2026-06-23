#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>
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
enum class SocketType { Float, Vec4, Sampler2D };

// Channel count -- what gets stored. Orthogonal to BitDepth.
enum class ChannelFormat : uint8_t { Mono, UV, RGB, RGBA };

// Bit depth -- how precise each channel is. The second axis of the format
// system (Substance Designer model). F8=unorm, F16/F32=float.
enum class BitDepth : uint8_t { F8, F16, F32 };

// How a node's bit depth is chosen -- inheritance mode (SD-style).
enum class DepthMode : uint8_t {
    Auto,        // graph default depth (from sidebar)
    MatchInput,  // upstream node's resolved depth
    Absolute,    // node.absolute_depth
};

// Full storage format descriptor. Channels and depth are independent axes;
// the full cross-product (Mono@F32, RGB@F8, etc.) is legal.
struct StorageFormat {
    ChannelFormat channels = ChannelFormat::RGBA;
    BitDepth      depth    = BitDepth::F16;

    bool operator==(const StorageFormat&) const = default;
};

// Resolve a StorageFormat to a concrete VkFormat. ID -> R32_UINT, Metadata -> RGBA32F.
VkFormat storage_format_to_vk(StorageFormat fmt);
// Bytes per pixel for budget checks.
uint32_t storage_format_bytes(StorageFormat fmt);
// GLSL storage image layout qualifier matching storage_format_to_vk exactly
// (e.g. "r8", "r16_sfloat", "rgba32f"). Eliminates the old bug where the
// shader declared r32f but the image was allocated as R16_SFLOAT.
std::string storage_format_glsl_qualifier(StorageFormat fmt);

// Deprecated shim -- calls storage_format_to_vk at fixed F16 depth.
// New code should pass StorageFormat. Kept during incremental migration.
inline VkFormat channel_to_vk_format(ChannelFormat fmt) {
    return storage_format_to_vk(StorageFormat{fmt, BitDepth::F16});
}


struct Socket {
    std::string name;   // "color", "height"
    SocketType type = SocketType::Vec4;
    ChannelFormat format = ChannelFormat::RGBA;
    // Float inputs: default_value is the only slot used (SSBO seed).
    // Vec4 inputs: default_vec4 is baked into GLSL when unconnected.
    float default_value = 0.0f;  // for SocketType::Float inputs
    std::array<float, 4> default_vec4 = {0.0f, 0.0f, 0.0f, 0.0f};
};


// PassKind classification (Stage 2). Lives in Graph.hpp (not
// PassPlan.hpp) so that NodeType can use it without creating a
// include cycle: NodeLibrary includes Graph, GraphIR includes both,
// PassPlan includes GraphIR. See 03_pass_kind.md §2.2 step 1 for
// the include-chain reasoning.
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

    // Stage 2: how this node participates in chain fusion. Set by
    // NodeRegistryLoader from the .node.json "pass_kind" key (defaults to
    // "compute"). Mirrored into ValidatedNode::pass_kind by the
    // validator, then into ComputePass::kind by GraphCompiler.
    // Stages 3-6 (chain find, chain emit, dispatch) consume this.
    PassKind pass_kind = PassKind::Compute;
};


struct NodeInstance {
    NodeId id = 0;
    std::string type_id;
    ChannelFormat format_override = ChannelFormat::RGBA;
    // Optional human-readable label. When non-empty, takes precedence over the
    // auto-generated "{type_id}_{id}" name in logs, error messages, and the
    // ResourceManager debug_name. Phase 1d feature.
    std::string debug_name;
    // Phase 1c: mute / bypass semantics.
    //   muted    — skip; downstream reads this node's input[0] source instead
    //              (validator rewires connections).
    //   bypassed — skip; downstream gets a clear-to-zero pass (compiler emits
    //              a no-op compute shader). Visible in the graph UI as
    //              "turned off" without removing the node.
    bool muted    = false;
    bool bypassed = false;
    // SD-style depth inheritance. Appended last to preserve aggregate-init
    // ordering for existing test/code sites using {id, type, fmt, name, m, b}.
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
