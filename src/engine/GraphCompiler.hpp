#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PassPlan.hpp"
#include <string>
#include <unordered_map>

namespace te {

struct CompileGraphResult {
    bool success = false;
    std::string error;
    PassPlan pass_plan;
    std::unordered_map<NodeId, int> param_base_slot;
    int total_param_floats = 0;

    // Retained for backward compat / debugging; no longer used by the engine.
    std::string glsl;
};

class GraphCompiler {
public:
    static CompileGraphResult compile(const GraphIR& ir,
                                      const NodeLibrary& lib,
                                      VkFormat output_format = VK_FORMAT_R32G32B32A32_SFLOAT);
};


} // namespace te