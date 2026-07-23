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
};

class GraphCompiler {
public:
    static CompileGraphResult compile(const GraphIR& ir,
                                      const NodeLibrary& lib);
};

std::string emit_node_shader(const ValidatedNode& validated_node,
                             const NodeType& type,
                             const ShaderVariantKey& key,
                             int param_base,
                             uint32_t input_count,
                             ChannelFormat format,
                             const std::vector<ResourceUUID>& input_resources);

} // namespace te