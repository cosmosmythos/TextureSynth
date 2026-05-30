#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeResource.hpp"
#include "engine/ShaderVariantKey.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace te {


enum class PassKind : uint8_t {
    Dispatch,     // normal compute pass
    ResourceBind, // image/external source — no vkCmdDispatch, only state tracked
};


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
    PassKind     kind               = PassKind::Dispatch;
    InputMode    input_mode         = InputMode::PreSampled;

    // Shader variant key — cache lookup is keyed by this, not source hash.
    ShaderVariantKey variant_key;

    // Partial re-execution: set by GraphCompiler, mutated at runtime.
    mutable bool dirty = true;
    mutable uint64_t last_executed_gen = 0;
    bool output_layout_is_general = false;
};


struct PassPlan {
    std::vector<ComputePass> passes;
    ResourceUUID final_output_resource = {};
};


} // namespace te