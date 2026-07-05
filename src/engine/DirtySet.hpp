#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>

namespace te {

class DirtySet {
public:
    void mark_node(uint64_t node_id) {
        dirty_nodes_.insert(node_id);
    }

    void mark_topology_change() {
        dirty_nodes_.clear();
        all_dirty_ = true;
    }

    void propagate(const std::unordered_map<uint64_t, std::vector<uint64_t>>& downstream) {
        if (all_dirty_) return;
        if (dirty_nodes_.empty()) return;

        std::unordered_set<uint64_t> visited = dirty_nodes_;
        std::vector<uint64_t> queue(dirty_nodes_.begin(), dirty_nodes_.end());

        size_t head = 0;
        while (head < queue.size()) {
            uint64_t cur = queue[head++];
            auto it = downstream.find(cur);
            if (it == downstream.end()) continue;
            for (uint64_t nxt : it->second) {
                if (visited.insert(nxt).second) {
                    dirty_nodes_.insert(nxt);
                    queue.push_back(nxt);
                }
            }
        }
    }

    bool is_dirty(uint64_t node_id) const {
        return all_dirty_ || dirty_nodes_.count(node_id) > 0;
    }

    bool any() const { return all_dirty_ || !dirty_nodes_.empty(); }

    void clear() {
        dirty_nodes_.clear();
        all_dirty_ = false;
    }

    const std::unordered_set<uint64_t>& nodes() const { return dirty_nodes_; }

private:
    std::unordered_set<uint64_t> dirty_nodes_;
    bool all_dirty_ = true;
};

} // namespace te
