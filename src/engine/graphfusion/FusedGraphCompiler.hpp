#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/GraphCompiler.hpp"

namespace te {

class FusedGraphCompiler {
public:
    static CompileGraphResult compile(const GraphIR& ir,
                                      const NodeLibrary& lib,
                                      NodeId active_node_id);

private:
    static constexpr uint32_t DEFAULT_REG_BUDGET = 48;
};

} // namespace te
