#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PassPlan.hpp"
#include <string>

namespace te::chain_shader {

// ---------------------------------------------------------------------------
// Stage 4.1 + 4.2: emit GLSL for a LINEAR chain (possibly with
// multi-input nodes).
//
// A chain in this stage is a sequence of PurePixel nodes where:
//   - the first node (head) has 0..N input sockets; every socket is
//     fed by an external sampled image (u_sampled[pc.in_sampled_slots[s]])
//   - every other node has 1..N input sockets; socket 0 is fed by the
//     previous chain node's output (Local{i-1}), and any additional
//     sockets are fed by external sampled images
//   - every input is SocketType::Vec4 (no Sampler2D, no Float)
//   - total external inputs across the whole chain <= MAX_PASS_INPUTS
//     (push-constant slot budget)
//
// The result is a single .comp shader that runs all nodes in a single
// dispatch. The push constant layout matches PassPushConstants so the
// existing Engine can dispatch chains using the same struct (Stage 6
// wires the actual vkCmdPushConstants; this stage only emits GLSL).
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
