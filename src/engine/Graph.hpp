#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>
#include <functional>


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
enum class ChannelFormat { Mono, UV, RGB, RGBA, ID, Metadata };


inline VkFormat channel_to_vk_format(ChannelFormat fmt) {
    if (fmt == ChannelFormat::Mono)      return VK_FORMAT_R16_SFLOAT;
    if (fmt == ChannelFormat::UV)        return VK_FORMAT_R16G16_SFLOAT;
    if (fmt == ChannelFormat::RGB)       return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (fmt == ChannelFormat::RGBA)      return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (fmt == ChannelFormat::ID)        return VK_FORMAT_R32_UINT;
    return VK_FORMAT_R16G16B16A16_SFLOAT;
}


struct Socket {
    std::string name;   // "color", "height"
    SocketType type = SocketType::Vec4;
    ChannelFormat format = ChannelFormat::RGBA;
    float default_value = 0.0f;  // for SocketType::Float inputs
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

    // True if this node's glsl_function returns the canonical noise/gradient
    // vec4(noise, grad.x, grad.y, 1) that the format post-process knows how
    // to fold into the requested output format. Set by NodeRegistryLoader
    // from the .node.json "format_sensitive" key. Only noise generators
    // (perlin, simplex, value, gabor, worley, white_noise) opt in; combiners
    // (blend, grayscale, invert, ...) and constant sources (color_const,
    // image) produce final colors and must NOT set this, or the Mono path
    // would collapse their output to a single channel.
    bool is_format_sensitive = false;

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
