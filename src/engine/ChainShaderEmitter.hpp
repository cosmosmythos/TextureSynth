#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PassPlan.hpp"
#include <string>

namespace te {

// Stage 4: emit GLSL for a LINEAR chain. A chain is a sequence of PurePixel nodes where head has 0..N external inputs, every other node has socket 0 fed by previous output (Local) and rest fed by external images. Every input must be Vec4. Total external inputs <= MAX_PASS_INPUTS. Returns Result with ok()=false on failure (caller falls back to per-node path).
struct Result {
    std::string source;        // GLSL source (empty if !ok())
    std::string error;         // human-readable diagnostic on failure
    uint32_t    external_inputs = 0;   // max in_sampled_slots index + 1 (0 for source-only)

    [[nodiscard]] constexpr bool ok() const noexcept { return error.empty(); }
};

// Emit a fused GLSL shader for a chain. Chain must be linear; Result.ok() false otherwise.
Result emit_linear(const Chain& chain, const GraphIR& ir,
                   const NodeLibrary& lib);

} // namespace te
