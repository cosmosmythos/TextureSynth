#include "engine/Engine.hpp"
#include "engine/Logging.hpp"
#include <algorithm>
#include <cstring>
#include <climits>
#include <unordered_set>

namespace te {

void Engine::seed_param_ssbo_defaults_() {
    for (uint32_t i = 0; i < PARAM_RING; ++i) {
        if (param_mapped_[i]) {
            std::memset(param_mapped_[i], 0, MAX_NODE_PARAMS * sizeof(float));
            vmaFlushAllocation(ctx_.allocator(), param_alloc_[i], 0, VK_WHOLE_SIZE);
        }
    }
    for (uint32_t ring = 0; ring < PARAM_RING; ++ring) {
        auto* dst = static_cast<float*>(param_mapped_[ring]);
        if (!dst) continue;

        for (NodeId node_id : current_ir_.eval_order) {
            auto bit = param_base_slot_.find(node_id);
            if (bit == param_base_slot_.end()) continue;

            auto validated_node_it = current_ir_.node_index.find(node_id);
            if (validated_node_it == current_ir_.node_index.end()) continue;
            const auto& validated_node = current_ir_.nodes[validated_node_it->second];

            const auto* type = node_lib_.find(validated_node.type_id);
            if (!type) continue;

            const int base = bit->second;
            for (size_t i = 0; i < type->params.size(); ++i) {
                const int slot = base + static_cast<int>(i);
                if (slot >= 0 && slot < static_cast<int>(MAX_NODE_PARAMS)) {
                    float v = type->params[i].default_value;
                    if (!std::isfinite(v)) v = 0.0f;
                    dst[slot] = v;
                }
            }
            uint32_t float_input_idx = 0;
            for (const auto& inp : type->inputs) {
                if (inp.type != SocketType::Float) continue;
                const int fslot = base + static_cast<int>(type->params.size()) + static_cast<int>(float_input_idx);
                if (fslot >= 0 && fslot < static_cast<int>(MAX_NODE_PARAMS)) {
                    float v = inp.default_value;
                    if (!std::isfinite(v)) v = 0.0f;
                    dst[fslot] = v;
                }
                ++float_input_idx;
            }
        }

        vmaFlushAllocation(ctx_.allocator(), param_alloc_[ring], 0, VK_WHOLE_SIZE);
    }
    param_write_idx_ = 0;
}


void Engine::update_node_params_by_id(NodeId node_id, const std::vector<float>& params) {
    TE_GUARD_READY(;);

    if (param_write_idx_ >= PARAM_RING) return;
    void* mapped = param_mapped_[param_write_idx_];
    if (!mapped) return;
    auto it = param_base_slot_.find(node_id);
    if (it == param_base_slot_.end()) {
        log_warn("update_node_params_by_id: unknown node id " + std::to_string(node_id));
        return;
    }
    const int base = it->second;
    if (base < 0) return;
    log_info("[param_write] node=" + std::to_string(node_id)
             + " base_slot=" + std::to_string(base)
             + " param_count=" + std::to_string(params.size())
             + " values[0]=" + std::to_string(params.empty() ? 0.f : params[0])
             + (params.size() > 1 ? " values[1]=" + std::to_string(params[1]) : ""));
    const size_t cap = (base >= 0 && (size_t)base < MAX_NODE_PARAMS)
                         ? (MAX_NODE_PARAMS - (size_t)base) : 0;
    if (params.size() > cap) {
        log_warn("update_node_params_by_id: truncating node " + std::to_string(node_id));
    }
    const size_t n = std::min(params.size(), cap);
    if (n > 0) {
        std::memcpy(static_cast<float*>(mapped) + base, params.data(), n * sizeof(float));
        vmaFlushAllocation(ctx_.allocator(), param_alloc_[param_write_idx_],
                           (VkDeviceSize)(base * sizeof(float)),
                           (VkDeviceSize)(n    * sizeof(float)));
    }
    param_dirty_ = true;
    any_pass_dirty_.store(true);
    dirty_set_.mark_node(node_id);
    mark_downstream_dirty_(node_id);
}


void Engine::update_node_params_by_name(NodeId node_id,
                                        const std::unordered_map<std::string, float>& kv) {
    TE_GUARD_READY(;);

    if (!param_mapped_[param_write_idx_]) return;

    auto it = param_base_slot_.find(node_id);
    if (it == param_base_slot_.end()) {
        log_warn("update_node_params_by_name: unknown node id "
                 + std::to_string(node_id));
        return;
    }
    const int base = it->second;

    const auto* validated_node = current_ir_.find(node_id);
    if (!validated_node) {
        log_warn("update_node_params_by_name: node " + std::to_string(node_id)
               + " not in current IR");
        return;
    }
    const auto* type = node_lib_.find(validated_node->type_id);
    if (!type) {
        log_warn("update_node_params_by_name: unknown type '" + validated_node->type_id
               + "' for node " + std::to_string(node_id));
        return;
    }

    auto* dst = static_cast<float*>(param_mapped_[param_write_idx_]);

    std::unordered_set<std::string> known;
    known.reserve(type->params.size());
    for (auto& p : type->params) known.insert(p.name);
    for (auto& [k, _v] : kv) {
        if (!known.count(k)) {
            log_warn("update_node_params_by_name: node " + std::to_string(node_id)
                   + " (type '" + type->id + "') has no param named '" + k
                   + "' -- ignoring.");
        }
    }

    int min_slot = INT_MAX, max_slot = -1;
    for (size_t i = 0; i < type->params.size(); ++i) {
        auto found = kv.find(type->params[i].name);
        if (found == kv.end()) continue;
        const int slot = base + static_cast<int>(i);
        if (slot < 0 || slot >= static_cast<int>(MAX_NODE_PARAMS)) {
            log_warn("update_node_params_by_name: slot " + std::to_string(slot)
                   + " out of range for param '" + type->params[i].name
                   + "' on node " + std::to_string(node_id)
                   + " -- DROPPED (bug: GraphCompiler should have rejected this graph).");
            continue;
        }
        dst[slot] = found->second;
        if (slot < min_slot) min_slot = slot;
        if (slot > max_slot) max_slot = slot;
    }
    if (max_slot >= 0) {
        const VkDeviceSize off  = (VkDeviceSize)(min_slot * sizeof(float));
        const VkDeviceSize size = (VkDeviceSize)((max_slot - min_slot + 1) * sizeof(float));
        vmaFlushAllocation(ctx_.allocator(), param_alloc_[param_write_idx_], off, size);
    }
    param_dirty_ = true;
    any_pass_dirty_.store(true);
    dirty_set_.mark_node(node_id);
    mark_downstream_dirty_(node_id);
}

} // namespace te
