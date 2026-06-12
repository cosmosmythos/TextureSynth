#include "engine/graphfusion/ActivePathTracer.hpp"
#include "engine/graphfusion/RegisterAllocator.hpp"
#include <algorithm>
#include <unordered_set>

namespace te {

dag::DAG<NodeId> ActivePathTracer::build_dag(const GraphIR& ir) {
    dag::DAG<NodeId>::NodeList nodes;
    nodes.reserve(ir.nodes.size());
    for (const auto& vn : ir.nodes) {
        nodes.push_back(vn.id);
    }

    dag::DAG<NodeId>::EdgeList edges;
    edges.reserve(ir.connections.size());
    for (const auto& conn : ir.connections) {
        edges.push_back({conn.src_node, conn.dst_node});
    }

    return dag::DAG<NodeId>(std::move(nodes), std::move(edges));
}

std::vector<NodeId> ActivePathTracer::topo_filter(
    const dag::DAG<NodeId>& dag,
    const std::vector<NodeId>& ancestors,
    NodeId active_node)
{
    auto topo = dag.topological_sort();
    std::unordered_set<NodeId> allowed(ancestors.begin(), ancestors.end());
    allowed.insert(active_node);

    std::vector<NodeId> result;
    for (NodeId n : topo) {
        if (allowed.count(n)) {
            result.push_back(n);
        }
    }
    return result;
}

ActivePath ActivePathTracer::trace(const GraphIR& ir, NodeId active_node_id,
                                   const NodeLibrary& lib) {
    ActivePath result;

    if (active_node_id == 0 || !ir.find(active_node_id)) {
        return result;
    }

    auto dag = build_dag(ir);

    auto ancestors = dag.ancestors_of(active_node_id);

    auto path = topo_filter(dag, ancestors, active_node_id);
    result.nodes = std::move(path);

    auto branch_pts = dag.branch_points(result.nodes);
    auto merge_pts = dag.merge_points(result.nodes);
    result.branch_points = std::move(branch_pts);
    result.merge_points = std::move(merge_pts);

    reg::RegisterAllocator alloc;
    std::vector<uint32_t> per_node_costs;
    per_node_costs.reserve(result.nodes.size());
    for (NodeId nid : result.nodes) {
        const auto* vn = ir.find(nid);
        if (!vn) { per_node_costs.push_back(5); continue; }
        const auto* type = lib.find(vn->type_id);
        if (!type) { per_node_costs.push_back(5); continue; }
        auto cost = reg::RegisterAllocator::estimate_cost(type->id);
        (void)alloc.add_node(cost);
        per_node_costs.push_back(cost.total());
    }
    result.estimated_registers = static_cast<std::uint32_t>(alloc.used());

    return result;
}

} // namespace te
