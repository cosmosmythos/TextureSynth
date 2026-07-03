#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace te::dag {

template <typename NodeId>
struct Edge {
    NodeId from{};
    NodeId to{};
};

template <
    typename NodeId,
    typename Hash = std::hash<NodeId>,
    typename KeyEqual = std::equal_to<NodeId>>
class DAG {
public:
    using node_type = NodeId;
    using edge_type = Edge<NodeId>;
    using NodeList = std::vector<NodeId>;
    using EdgeList = std::vector<edge_type>;
    using size_type = std::size_t;

    DAG() = default;

    DAG(NodeList nodes, EdgeList edges)
        : nodes_(std::move(nodes))
        , edges_(std::move(edges))
    {
        rebuild_indexes();
    }

    [[nodiscard]] const NodeList& nodes() const noexcept { return nodes_; }
    [[nodiscard]] const EdgeList& edges() const noexcept { return edges_; }

    [[nodiscard]] bool has_node(const NodeId& id) const noexcept {
        return node_index_.find(id) != node_index_.end();
    }

    [[nodiscard]] bool has_edge(const NodeId& from, const NodeId& to) const noexcept {
        const auto it = outgoing_.find(from);
        if (it == outgoing_.end()) {
            return false;
        }
        const auto& successors = it->second;
        return std::find(successors.begin(), successors.end(), to) != successors.end();
    }

    [[nodiscard]] const NodeList& predecessors(const NodeId& id) const {
        static const NodeList empty;
        const auto it = incoming_.find(id);
        return it != incoming_.end() ? it->second : empty;
    }

    [[nodiscard]] const NodeList& successors(const NodeId& id) const {
        static const NodeList empty;
        const auto it = outgoing_.find(id);
        return it != outgoing_.end() ? it->second : empty;
    }

    [[nodiscard]] std::uint32_t in_degree(const NodeId& id) const {
        const auto it = incoming_.find(id);
        return it != incoming_.end() ? static_cast<std::uint32_t>(it->second.size()) : 0u;
    }

    [[nodiscard]] std::uint32_t out_degree(const NodeId& id) const {
        const auto it = outgoing_.find(id);
        return it != outgoing_.end() ? static_cast<std::uint32_t>(it->second.size()) : 0u;
    }

    template <typename Predicate>
    [[nodiscard]] NodeList filter(Predicate&& pred) const {
        NodeList result;
        result.reserve(nodes_.size());

        for (const auto& node : nodes_) {
            if (std::invoke(pred, node)) {
                result.push_back(node);
            }
        }

        return result;
    }

    [[nodiscard]] NodeList reachable_from(const NodeId& start) const {
        return breadth_first_search(start, false);
    }

    [[nodiscard]] NodeList bfs_from(const NodeId& start) const {
        return reachable_from(start);
    }

    [[nodiscard]] NodeList ancestors_of(const NodeId& id) const {
        if (!has_node(id)) {
            return {};
        }

        NodeList result;
        result.reserve(nodes_.size());

        std::unordered_set<NodeId, Hash, KeyEqual> visited;
        visited.reserve(nodes_.size());

        std::queue<NodeId> pending;
        // Seed with predecessors of id, not id itself
        const auto it = incoming_.find(id);
        if (it != incoming_.end()) {
            for (const auto& pred : it->second) {
                if (visited.insert(pred).second) {
                    pending.push(pred);
                }
            }
        }

        while (!pending.empty()) {
            const NodeId current = pending.front();
            pending.pop();
            result.push_back(current);

            const auto pit = incoming_.find(current);
            if (pit == incoming_.end()) {
                continue;
            }

            for (const auto& next : pit->second) {
                if (visited.insert(next).second) {
                    pending.push(next);
                }
            }
        }

        return result;
    }

    [[nodiscard]] DAG subgraph(const NodeList& subset) const {
        std::unordered_set<NodeId, Hash, KeyEqual> allowed(subset.begin(), subset.end());

        NodeList sub_nodes;
        sub_nodes.reserve(subset.size());
        for (const auto& node : nodes_) {
            if (allowed.find(node) != allowed.end()) {
                sub_nodes.push_back(node);
            }
        }

        EdgeList sub_edges;
        sub_edges.reserve(edges_.size());
        for (const auto& edge : edges_) {
            if (allowed.find(edge.from) != allowed.end() &&
                allowed.find(edge.to) != allowed.end()) {
                sub_edges.push_back(edge);
            }
        }

        return DAG(std::move(sub_nodes), std::move(sub_edges));
    }

    [[nodiscard]] NodeList branch_points(const NodeList& subset) const {
        return degree_points(subset, true);
    }

    [[nodiscard]] NodeList merge_points(const NodeList& subset) const {
        return degree_points(subset, false);
    }

private:
    void rebuild_indexes() {
        node_index_.clear();
        incoming_.clear();
        outgoing_.clear();

        node_index_.reserve(nodes_.size());
        incoming_.reserve(nodes_.size());
        outgoing_.reserve(nodes_.size());

        for (size_type i = 0; i < nodes_.size(); ++i) {
            const auto& node = nodes_[i];
            node_index_[node] = i;
            incoming_.try_emplace(node);
            outgoing_.try_emplace(node);
        }

        for (const auto& edge : edges_) {
            if (!has_node(edge.from) || !has_node(edge.to)) {
                throw std::invalid_argument("DAG edge references a node not present in the node set.");
            }
            outgoing_[edge.from].push_back(edge.to);
            incoming_[edge.to].push_back(edge.from);
        }
    }

    [[nodiscard]] NodeList breadth_first_search(const NodeId& start, bool follow_predecessors) const {
        if (!has_node(start)) {
            return {};
        }

        NodeList result;
        result.reserve(nodes_.size());

        std::unordered_set<NodeId, Hash, KeyEqual> visited;
        visited.reserve(nodes_.size());

        std::queue<NodeId> pending;
        pending.push(start);
        visited.insert(start);

        while (!pending.empty()) {
            const NodeId current = pending.front();
            pending.pop();
            result.push_back(current);

            const auto& adjacency = follow_predecessors ? incoming_ : outgoing_;
            const auto it = adjacency.find(current);
            if (it == adjacency.end()) {
                continue;
            }

            for (const auto& next : it->second) {
                if (visited.insert(next).second) {
                    pending.push(next);
                }
            }
        }

        return result;
    }

    [[nodiscard]] NodeList degree_points(const NodeList& subset, bool outgoing) const {
        std::unordered_set<NodeId, Hash, KeyEqual> allowed(subset.begin(), subset.end());
        NodeList result;

        for (const auto& node : subset) {
            std::size_t count = 0;
            const auto adjacency = outgoing ? successors(node) : predecessors(node);

            for (const auto& connected : adjacency) {
                if (allowed.find(connected) != allowed.end()) {
                    ++count;
                    if (count > 1U) {
                        result.push_back(node);
                        break;
                    }
                }
            }
        }

        return result;
    }

    NodeList nodes_;
    EdgeList edges_;
    std::unordered_map<NodeId, size_type, Hash, KeyEqual> node_index_;
    std::unordered_map<NodeId, NodeList, Hash, KeyEqual> incoming_;
    std::unordered_map<NodeId, NodeList, Hash, KeyEqual> outgoing_;
};

} // namespace te::dag
