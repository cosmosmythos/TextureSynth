#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cstdint>

namespace te {

using ChainId = uint32_t;   // NodeId is defined in Graph.hpp

// Owns the "which passes need to run this frame" decision. Decoupled from PassPlan for unit-testability.
class DirtySet {
public:
    void mark_node(uint64_t node_id) {
        const bool new_ = dirty_nodes_.insert(node_id).second;
        if (!new_) return;
        // Invalidate every chain that contains this node.
        for (auto& kv : chain_members_) {
            for (NodeId n : kv.second) {
                if (n == node_id) {
                    chain_dirty_cache_.erase(kv.first);
                    break;
                }
            }
        }
    }

    void mark_topology_change() {
        dirty_nodes_.clear();
        all_dirty_ = true;
        chain_dirty_cache_.clear();
    }

    // Expand dirty set to include all downstream consumers of marked nodes.
    void propagate(const std::unordered_map<uint64_t, std::vector<uint64_t>>& downstream) {
        if (all_dirty_) return;  // everything is already dirty
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
                    mark_node(nxt);  // routes through mark_node so chain cache invalidates
                    queue.push_back(nxt);
                }
            }
        }
    }

    bool is_dirty(uint64_t node_id) const {
        return all_dirty_ || dirty_nodes_.count(node_id) > 0;
    }

    // Stage 6: chain-level dirty query. True if chain unknown (conservative), all_dirty_, or any member in dirty_nodes_. Cached per chain id; invalidated by mark_node, mark_topology_change, clear, set_chain_membership.
    bool is_chain_dirty(ChainId c) const {
        if (all_dirty_) return true;
        auto cit = chain_dirty_cache_.find(c);
        if (cit != chain_dirty_cache_.end()) return cit->second;
        auto it = chain_members_.find(c);
        if (it == chain_members_.end()) {
            chain_dirty_cache_[c] = true;   // unknown -> run it
            return true;
        }
        for (NodeId n : it->second) {
            if (dirty_nodes_.count(n) > 0) {
                chain_dirty_cache_[c] = true;
                return true;
            }
        }
        chain_dirty_cache_[c] = false;
        return false;
    }

    bool any() const { return all_dirty_ || !dirty_nodes_.empty(); }

    void clear() {
        dirty_nodes_.clear();
        all_dirty_ = false;
        chain_dirty_cache_.clear();
    }

    // Stage 6: set chain membership (once per graph compile). Does NOT reset all_dirty_ (topology changes use mark_topology_change()).
    void set_chain_membership(std::unordered_map<ChainId, std::vector<NodeId>> m) {
        chain_members_ = std::move(m);
        chain_dirty_cache_.clear();
    }

    const std::unordered_set<uint64_t>& nodes() const { return dirty_nodes_; }

private:
    std::unordered_set<uint64_t> dirty_nodes_;
    bool all_dirty_ = true;  // first frame

    // Stage 6: chain membership + per-chain dirty cache.
    std::unordered_map<ChainId, std::vector<NodeId>> chain_members_;
    mutable std::unordered_map<ChainId, bool> chain_dirty_cache_;
};

} // namespace te
