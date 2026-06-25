#pragma once
#include "engine/GraphIR.hpp"   // transitively brings in PassKind from Graph.hpp
#include "engine/NodeResource.hpp"
#include "engine/ShaderVariantKey.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_set>

namespace te {
// PassKind enum lives in Graph.hpp to avoid cycles.


enum class InputMode : uint8_t {
    PreSampled,   // bound as storage images, loaded via imageLoad at current UV
    Sampler,      // bound as sampler2D (for blur/warp). Reserved for Phase 7.
};


struct ComputePass {
    NodeId       node_id            = 0;
    std::string  type_id;                       // shader signature key
    std::vector<ResourceUUID> input_resources;
    std::vector<ChannelFormat> input_formats;    // format per input socket
	std::vector<ResourceUUID> output_resources;  // multi-output nodes
    int          param_base_slot    = 0;
    uint32_t     input_socket_count = 0;
    std::string  shader_glsl;
    PassKind     kind               = PassKind::Compute;
    InputMode    input_mode         = InputMode::PreSampled;

    // Cache lookup is keyed by this, not source hash.
    ShaderVariantKey variant_key;

    // Phase 1c: mirrors ValidatedNode::bypassed. Retained in plan (subgraph/resource layout stable), executor emits clear-to-zero instead of normal shader. Bypassed = logical pass producing known-empty output, not removal.
    bool bypassed = false;
};


// A Chain is a maximal run of adjacent Compute nodes fused into a single compute shader. Stage 3 builds nodes/params/input-output layout. Stage 4 populates glsl. Stage 5 populates variant_key. Stage 6 consumes them.
struct Chain {
    std::vector<NodeId>   nodes;            // in topological order; first = head, last = tail
    std::vector<uint32_t> param_offsets;    // SSBO offset of each node's params; size == nodes.size()
    std::vector<uint32_t> param_global_slots; // each node's global param_base_slot
    int                   param_base_slot = 0;  // SSBO slot of nodes[0]
    uint32_t              total_inputs   = 0;   // sum across nodes (descriptor layout)
    uint32_t              total_outputs  = 1;   // 1 in Phase 1; multi-output nodes are barriers (singleton chains)
    uint32_t              total_params   = 0;   // sum across nodes (SSBO slice width)
    // Stage 4.2: per-node bitmask of which input sockets are cross-group texture inputs.
    // Size == nodes.size(). Bit s = 1 means socket s is external (ExtSrc).
    // Used for FusedVariantKey — encodes topology, not just count.
    std::vector<uint32_t>   external_socket_masks;

    // Stage 4.2: per-socket internal-producer local_index (flat, node-major).
    // Length = sum of input_counts. UINT32_MAX = not an internal RegSrc.
    // Used for FusedVariantKey — encodes WHICH in-chain producer feeds WHICH socket.
    std::vector<uint32_t>   internal_producer_indices;

    // Filled by Stage 4 (emit_chain_shader). Empty for singleton/oversized chains (falls back to per-node).
    std::string           glsl;

    // Filled by Stage 5 (FusedVariantKey). Cache lookup in Stage 6 uses this key; per-node key is fallback.
    FusedVariantKey       variant_key;

    // Filled by Stage 5 (FusedVariantKey). Empty for now; the cache
    // lookup uses the per-node key until Stage 5 lands.

    // Phase 1c: mirrors ComputePass::bypassed. If any node is bypassed, the whole chain is bypassed (clear-to-zero).
    bool                  bypassed = false;

    // Multi-pass: sub-pass GLSL and variant keys for nodes with pass_count > 1.
    // Single-pass chains: sub_pass_count == 0 (use legacy single glsl/pipeline).
    std::vector<std::string>      sub_pass_glsl;
    std::vector<ShaderVariantKey> sub_pass_variant_keys;
    uint32_t sub_pass_count     = 0;  // 0 = legacy single-pipeline chain
    uint32_t intermediate_count = 0;  // temp images between sub-passes
};


struct PassPlan {
    std::vector<ComputePass> passes;
    std::vector<Chain>       chains;     // Stage 3: superset info for fused runtime
    ResourceUUID final_output_resource = {};

    // Stage 6 aliasing: per-pass chain index (UINT32_MAX = not in chain).
    // Precomputed here rather than re-derived in Engine::populate_chains_.
    std::vector<uint32_t> chain_index_of_pass;

    struct ResourceLifetime {
        uint32_t first_pass = UINT32_MAX;  // pass index that writes this resource
        uint32_t last_pass  = 0;           // last pass index that reads it
    };
    std::unordered_map<ResourceUUID, ResourceLifetime, ResourceUUIDHash> lifetimes;
    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> color_classes;
    std::unordered_set<ResourceUUID, ResourceUUIDHash> active_resources;
};


} // namespace te