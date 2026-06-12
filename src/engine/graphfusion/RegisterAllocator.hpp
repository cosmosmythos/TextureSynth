#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace te::reg {

struct NodeRegCost {
    std::uint32_t output_regs = 1;
    std::uint32_t param_regs = 0;
    std::uint32_t temp_regs = 0;
    std::uint32_t sampler_regs = 0;
    std::uint32_t input_regs = 1;

    [[nodiscard]] std::uint32_t total() const noexcept {
        return output_regs + param_regs + temp_regs + sampler_regs + input_regs;
    }
};

class RegisterAllocator {
public:
    static constexpr std::uint32_t DEFAULT_BUDGET = 48;

    explicit RegisterAllocator(std::uint32_t budget = DEFAULT_BUDGET) noexcept
        : budget_(budget) {}

    void register_type(std::string type_id, NodeRegCost cost) {
        type_costs_[std::move(type_id)] = cost;
    }

    [[nodiscard]] bool has_registered_type(const std::string& type_id) const {
        return type_costs_.find(type_id) != type_costs_.end();
    }

    [[nodiscard]] static NodeRegCost estimate_cost(const std::string& type_id) {
        if (type_id.find("texture") != std::string::npos ||
            type_id.find("sample") != std::string::npos) {
            return {1, 0, 2, 2, 1};
        }

        if (type_id.find("math") != std::string::npos ||
            type_id.find("blend") != std::string::npos) {
            return {1, 0, 1, 0, 1};
        }

        return {1, 0, 2, 1, 1};
    }

    [[nodiscard]] NodeRegCost cost_for(const std::string& type_id) const {
        if (const auto it = type_costs_.find(type_id); it != type_costs_.end()) {
            return it->second;
        }
        return estimate_cost(type_id);
    }

    [[nodiscard]] bool would_exceed(const std::string& type_id) const {
        return would_exceed(cost_for(type_id));
    }

    [[nodiscard]] bool add_node(const std::string& type_id) {
        return add_node(cost_for(type_id));
    }

    [[nodiscard]] bool would_exceed(NodeRegCost cost) const {
        return saturating_add(used_, cost.total()) > budget_;
    }

    [[nodiscard]] bool add_node(NodeRegCost cost) {
        used_ = saturating_add(used_, cost.total());
        ++node_count_;
        return used_ <= budget_;
    }

    [[nodiscard]] std::uint64_t used() const noexcept { return used_; }

    [[nodiscard]] std::uint64_t remaining() const noexcept {
        return used_ >= budget_ ? 0ULL : static_cast<std::uint64_t>(budget_ - used_);
    }

    [[nodiscard]] std::uint32_t node_count() const noexcept { return node_count_; }

    [[nodiscard]] std::uint32_t budget() const noexcept { return static_cast<std::uint32_t>(budget_); }

    void reset() noexcept {
        used_ = 0;
        node_count_ = 0;
    }

    [[nodiscard]] std::uint32_t pressure_percent() const noexcept {
        if (budget_ == 0U) {
            return 100U;
        }
        const std::uint64_t pct = (used_ * 100ULL) / budget_;
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(pct, 100ULL));
    }

    [[nodiscard]] bool is_critical() const noexcept {
        return pressure_percent() > 80U;
    }

private:
    [[nodiscard]] static std::uint64_t saturating_add(std::uint64_t lhs, std::uint64_t rhs) noexcept {
        const std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
        if (lhs > limit - rhs) {
            return limit;
        }
        return lhs + rhs;
    }

    std::uint64_t budget_;
    std::uint64_t used_ = 0;
    std::uint32_t node_count_ = 0;
    std::unordered_map<std::string, NodeRegCost> type_costs_;
};

} // namespace te::reg
