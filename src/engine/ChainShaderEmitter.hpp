#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PassPlan.hpp"
#include <string>

namespace te::chain_shader {

// ---------------------------------------------------------------------------
// Stage 4.1: emit GLSL for a LINEAR chain.
//
// A "linear" chain (the Stage 4.1 subset) is a sequence of single-input
// PurePixel nodes where:
//   - the first node has 0 inputs (source) or 1 input from outside the chain
//   - every other node has exactly 1 input, which is the previous node's output
//   - every input is SocketType::Vec4 (no Sampler2D, no Float -- Stage 4.1)
//
// The result is a single .comp shader that runs all nodes in a single
// dispatch. The push constant layout matches PassPushConstants so the
// existing Engine can dispatch chains using the same struct (Stage 6
// wires the actual vkCmdPushConstants; Stage 4.1 only emits GLSL).
//
// Returns Result with ok()=false on any failure: caller should fall back
// to the per-node path.
// ---------------------------------------------------------------------------
struct Result {
    std::string source;        // GLSL source (empty if !ok())
    std::string error;         // human-readable diagnostic on failure
    uint32_t    external_inputs = 0;   // = max in_sampled_slots index used + 1
                                       // (0 for source-only chains)

    [[nodiscard]] constexpr bool ok() const noexcept { return error.empty(); }
};

// Emit a fused GLSL shader for a chain. The chain must be linear per the
// contract above; Result.ok() will be false otherwise.
Result emit_linear(const Chain& chain, const GraphIR& ir,
                   const NodeLibrary& lib);

} // namespace te::chain_shader
