#pragma once

#include "engine/GraphIR.hpp"
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>

namespace te {

// Filters ir.eval_order to nodes reachable backward from active_node.
// Replaces the deleted ActivePathTracer::trace() in test code.
inline std::vector<NodeId> trace_active(const GraphIR& ir, NodeId active_node) {
    std::unordered_map<NodeId, std::vector<NodeId>> consumers;
    for (const auto& c : ir.connections)
        consumers[c.dst_node].push_back(c.src_node);

    std::unordered_set<NodeId> ancestors;
    std::queue<NodeId> q;
    q.push(active_node);
    ancestors.insert(active_node);
    while (!q.empty()) {
        NodeId cur = q.front(); q.pop();
        auto it = consumers.find(cur);
        if (it == consumers.end()) continue;
        for (NodeId src : it->second)
            if (ancestors.insert(src).second)
                q.push(src);
    }

    std::vector<NodeId> result;
    for (NodeId n : ir.eval_order)
        if (ancestors.count(n))
            result.push_back(n);
    return result;
}

} // namespace te
