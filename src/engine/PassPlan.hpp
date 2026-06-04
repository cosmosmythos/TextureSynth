#pragma once
#include "engine/GraphIR.hpp"   // transitively brings in PassKind from Graph.hpp
#include "engine/NodeResource.hpp"
#include "engine/ShaderVariantKey.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace te {


// PassKind enum itself lives in Graph.hpp so NodeType can use it without
// a cycle. See 03_pass_kind.md §2.2 step 1 for the include-chain reasoning.


enum class InputMode : uint8_t {
    PreSampled,   // bound as storage images, loaded via imageLoad at current UV
    Sampler,      // bound as sampler2D (for blur/warp). Reserved for Phase 7.
};


struct ComputePass {
    NodeId       node_id            = 0;
    std::string  type_id;                       // shader signature key
    std::vector<ResourceUUID> input_resources;
	std::vector<ResourceUUID> output_resources;  // multi-output nodes
    int          param_base_slot    = 0;
    uint32_t     input_socket_count = 0;
    std::string  shader_glsl;
    PassKind     kind               = PassKind::Dispatch;   // legacy binary (executor uses this)
    PassKind     pass_kind          = PassKind::PurePixel;   // Stage 2 classification (stages 3-6 consume)
    InputMode    input_mode         = InputMode::PreSampled;

    // Shader variant key -- cache lookup is keyed by this, not source hash.
    ShaderVariantKey variant_key;

    // Partial re-execution: set by GraphCompiler, mutated at runtime.
    mutable bool dirty = true;
    mutable uint64_t last_executed_gen = 0;
    bool output_layout_is_general = false;

    // Phase 1c: mirrors ValidatedNode::bypassed. A bypassed pass is
    // retained in the plan (so the active-subgraph and resource layout
    // stay stable when the user toggles the flag) but the executor
    // will emit a clear-to-zero dispatch instead of the node's normal
    // shader. Bypassed = "logical pass that produces a known-empty
    // output", not a removal.
    bool bypassed = false;
};


// A Chain is a maximal run of adjacent PurePixel nodes that can be
// fused into a single compute shader. The compiler builds these in
// GraphCompiler::compile (Stage 3). Stage 4 (emit_chain_shader)
// populates `glsl`. Stage 5 (FusedVariantKey) populates `variant_key`.
// Stage 6 (per-chain dispatch) consumes them. Stage 3 only fills
// `nodes`, `param_offsets`, `param_base_slot`, `total_params`,
// `total_inputs`, and `total_outputs`.
//
// See 04_chains.md §3.1 for the struct spec, and
// 06_chain_finding_research.md for the algorithm rationale.
struct Chain {
    std::vector<NodeId>   nodes;            // in topological order; first = head, last = tail
    std::vector<uint32_t> param_offsets;    // SSBO offset of each node's params; size == nodes.size()
    int                   param_base_slot = 0;  // SSBO slot of nodes[0]
    uint32_t              total_inputs   = 0;   // sum across nodes (descriptor layout)
    uint32_t              total_outputs  = 1;   // 1 in Phase 1; multi-output nodes are barriers (singleton chains)
    uint32_t              total_params   = 0;   // sum across nodes (SSBO slice width)
    // Stage 4.2: # of pc.in_sampled_slots[] indices this chain's shader
    // consumes (= max slot index used + 1). 0 for source-only chains.
    // Populated by GraphCompiler from chain_shader::Result::external_inputs.
    uint32_t              external_inputs = 0;

    // Filled by Stage 4 (emit_chain_shader). Empty for singleton or
    // oversized chains; the runtime then falls back to the per-node path.
    std::string           glsl;

    // Filled by Stage 5 (FusedVariantKey). The cache lookup in Stage 6
    // uses this key; per-node ShaderVariantKey is used for the fallback path.
    FusedVariantKey       variant_key;

    // Filled by Stage 5 (FusedVariantKey). Empty for now; the cache
    // lookup uses the per-node key until Stage 5 lands.

    // Phase 1c: mirrors ComputePass::bypassed. If any node in the chain
    // is bypassed, the chain is bypassed and the executor emits a
    // clear-to-zero pass for the whole chain.
    bool                  bypassed = false;
};


struct PassPlan {
    std::vector<ComputePass> passes;
    std::vector<Chain>       chains;     // Stage 3: superset info for fused runtime
    ResourceUUID final_output_resource = {};
};


} // namespace te