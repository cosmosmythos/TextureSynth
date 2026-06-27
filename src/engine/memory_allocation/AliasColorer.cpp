#include "engine/memory_allocation/AliasColorer.hpp"
#include "engine/Graph.hpp"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace te::memory_allocation {

bool AliasColorer::formats_compatible(StorageFormat a, StorageFormat b) {
    return storage_format_bytes(a) == storage_format_bytes(b);
}

AliasColoringResult AliasColorer::compute(
    const std::vector<ComputePass>& passes,
    const ResourceUUID& final_output,
    const GraphIR& ir,
    const NodeLibrary& lib)
{
    // 1. Compute pass-level lifetimes.
    std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash> lifetimes;
    for (uint32_t i = 0; i < (uint32_t)passes.size(); ++i) {
        const auto& pass = passes[i];
        for (const auto& rid : pass.output_resources) {
            auto& lt = lifetimes[rid];
            if (lt.first_pass == UINT32_MAX) lt.first_pass = i;
            lt.last_pass = i;
        }
        for (const auto& rid : pass.input_resources) {
            if (rid.node_id == 0) continue;
            auto& lt = lifetimes[rid];
            lt.last_pass = i;
            if (lt.first_pass == UINT32_MAX) lt.first_pass = 0;
        }
    }
    auto& fo_lt = lifetimes[final_output];
    fo_lt.first_pass = 0;
    fo_lt.last_pass = UINT32_MAX;

    // 2. Resolve storage format for each resource.
    std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash> formats;
    for (const auto& vn : ir.nodes) {
        const auto* type = lib.find(vn.type_id);
        if (!type) continue;
        const uint32_t num_outputs = std::max(1u, (uint32_t)type->outputs.size());
        for (uint32_t output_id = 0; output_id < num_outputs; ++output_id) {
            ResourceUUID rid{vn.id, output_id};
            formats[rid] = resolve_node_storage(vn, lib, output_id);
        }
    }

    return compute_from_lifetimes(lifetimes, formats);
}

AliasColoringResult AliasColorer::compute_from_lifetimes(
    const std::unordered_map<ResourceUUID, AliasLifetime, ResourceUUIDHash>& lifetimes,
    const std::unordered_map<ResourceUUID, StorageFormat, ResourceUUIDHash>& formats)
{
    AliasColoringResult result;
    result.lifetimes = lifetimes;

    // 1. Collect aliasable resources: multi-pass, not final output, bounded lifetime.
    struct Aliasable {
        ResourceUUID rid;
        uint32_t first;
        uint32_t last;
        StorageFormat format;
    };
    std::vector<Aliasable> items;
    for (const auto& kv : lifetimes) {
        if (kv.second.first_pass == kv.second.last_pass) continue;
        if (kv.second.last_pass == UINT32_MAX) continue;
        auto fit = formats.find(kv.first);
        StorageFormat fmt = (fit != formats.end()) ? fit->second : StorageFormat{};
        items.push_back({kv.first, kv.second.first_pass, kv.second.last_pass, fmt});
    }

    // 2. Group by format (same bytes/pixel).
    // Key = bytes_per_pixel, Value = indices into items vector.
    std::unordered_map<uint32_t, std::vector<size_t>> format_groups;
    for (size_t i = 0; i < items.size(); ++i) {
        uint32_t bpp = storage_format_bytes(items[i].format);
        format_groups[bpp].push_back(i);
    }

    // 3. Within each format group, run greedy first-fit interval coloring.
    uint32_t next_color = 1;
    for (auto& kv : format_groups) {
        auto& indices = kv.second;

        // Sort by (first, last).
        std::sort(indices.begin(), indices.end(),
            [&](size_t a, size_t b) {
                if (items[a].first != items[b].first)
                    return items[a].first < items[b].first;
                return items[a].last < items[b].last;
            });

        // Greedy first-fit.
        std::vector<uint32_t> color_end;
        for (size_t idx : indices) {
            const auto& item = items[idx];
            bool found = false;
            for (uint32_t c = 0; c < (uint32_t)color_end.size(); ++c) {
                if (color_end[c] < item.first) {
                    result.color_classes[item.rid] = c + 1;
                    color_end[c] = item.last;
                    found = true;
                    break;
                }
            }
            if (!found) {
                result.color_classes[item.rid] = next_color;
                color_end.push_back(item.last);
                ++next_color;
            }
        }
    }

    result.groups_created = next_color - 1;
    return result;
}

} // namespace te::memory_allocation
