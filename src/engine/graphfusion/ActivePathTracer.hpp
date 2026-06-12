#pragma once

#include "engine/graphfusion/DAG.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include <vector>

namespace te {

struct ActivePath {
    std::vector<NodeId> nodes;
    std::vector<NodeId> branch_points;
    std::vector<NodeId> merge_points;
    std::uint32_t estimated_registers = 0;
};

class ActivePathTracer {
public:
    static ActivePath trace(const GraphIR& ir, NodeId active_node_id,
                            const NodeLibrary& lib);

private:
    static dag::DAG<NodeId> build_dag(const GraphIR& ir);
    static std::vector<NodeId> topo_filter(const dag::DAG<NodeId>& dag,
                                           const std::vector<NodeId>& ancestors,
                                           NodeId active_node);
};

} // namespace te
