#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/graphfusion/ActivePathTracer.hpp"
#include <string>

namespace te {

struct FusedResult {
    std::string source;
    std::string error;
    uint32_t    external_inputs = 0;

    [[nodiscard]] constexpr bool ok() const noexcept { return error.empty(); }
};

FusedResult emit_fused_subgraph(
    const ActivePath& path,
    const GraphIR& ir,
    const NodeLibrary& lib,
    uint32_t param_base_slot);

} // namespace te
