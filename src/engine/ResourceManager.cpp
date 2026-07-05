#include "engine/ResourceManager.hpp"
#include "engine/VulkanContext.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include <algorithm>


namespace te {


constexpr uint32_t RETIRE_FRAMES = 4; // MAX_FRAMES_IN_FLIGHT + safety


size_t ResourceManager::pixel_bytes_(VkFormat f) const {
    const uint32_t bytes = vk_format_bytes(f);
    return bytes == 0 ? 4 : bytes;
}


bool ResourceManager::create_image_(VulkanContext& ctx, NodeResource& r,
                                    uint32_t w, uint32_t h,
                                    const std::string& dbg,
                                    VmaPool pool, bool can_alias) {
    r.extent = {w, h, 1};
    r.debug_name = dbg;
    r.layout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format    = r.format;
    ici.extent    = r.extent;
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples   = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
    ici.usage     = VK_IMAGE_USAGE_STORAGE_BIT
                  | VK_IMAGE_USAGE_SAMPLED_BIT
                  | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                  | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (can_alias) ici.flags |= VK_IMAGE_CREATE_ALIAS_BIT;

    std::string usage_error;
    if (!ctx.format_supports_image_usage(r.format, ici.usage, &usage_error)) {
        log_error("ResourceManager: " + usage_error + " for " + dbg);
        return false;
    }

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.pool = pool;
    if (can_alias) aci.flags |= VMA_ALLOCATION_CREATE_CAN_ALIAS_BIT;
    if (vmaCreateImage(ctx.allocator(), &ici, &aci, &r.image, &r.alloc, nullptr) != VK_SUCCESS) {
        log_error("ResourceManager: vmaCreateImage failed for " + dbg);
        return false;
    }
    ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)r.image, dbg);
    // VMA's own name mechanism (independent of vkSetDebugUtilsObjectNameEXT). Populated via vmaGetAllocationInfo(alloc).pName, consumed by vmaBuildStatsString JSON dump.
    vmaSetAllocationName(ctx.allocator(), r.alloc, dbg.c_str());

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image    = r.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = r.format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(ctx.device(), &vci, nullptr, &r.view) != VK_SUCCESS) {
        log_error("ResourceManager: view creation failed for " + dbg);
        vmaDestroyImage(ctx.allocator(), r.image, r.alloc);
        r.image = VK_NULL_HANDLE;
        r.alloc = nullptr;
        return false;
    }
    ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)r.view, dbg + "_view");
    return true;
}


void ResourceManager::destroy_image_(VulkanContext& ctx, NodeResource& r) {
    if (r.view)  { vkDestroyImageView(ctx.device(), r.view, nullptr); r.view = VK_NULL_HANDLE; }
    if (r.image) {
        if (r.alloc) {
            // Primary: vmaDestroyImage destroys the VkImage AND frees the
            // shared VmaAllocation. Siblings in the same alias group are
            // guaranteed to have been destroyed first (see tick/shutdown
            // two-pass order).
            vmaDestroyImage(ctx.allocator(), r.image, r.alloc);
        } else {
            // Sibling: image is bound to another NodeResource's VmaAllocation.
            // Only destroy the VkImage; the alloc will be freed by the primary.
            vkDestroyImage(ctx.device(), r.image, nullptr);
        }
        r.image = VK_NULL_HANDLE;
        r.alloc = nullptr;
    }
}


bool ResourceManager::allocate_for_graph(VulkanContext& ctx,
                                         const GraphIR& ir, const NodeLibrary& lib,
                                         uint32_t width, uint32_t height,
                                         VkFormat default_format,
                                         std::string* error,
                                         const std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash>* color_classes,
                                         const std::unordered_set<ResourceUUID, ResourceUUIDHash>* active_resources) {
    retire_all(ctx);
    // alias_pools_ holds metadata only (no VmaPools anymore); the actual
    // VmaAllocations are owned by primary NodeResources and freed via
    // vmaDestroyImage in destroy_image_().
    alias_pools_.clear();

    // -- Stage A: gather resource info + budget check ---------------------
    //
    // Budget check counts bytes for EVERY node (including nodes whose type
    // is missing from the library) so a misconfigured node set can't sneak
    // past the gate. The actual allocation loop below skips untyped nodes
    // (we can't determine their format) but still rejects the graph as a
    // whole.
    size_t total = 0;
    for (const auto& vn : ir.nodes) {
        const auto* type = lib.find(vn.type_id);
        const uint32_t num_outputs = type ? std::max(1u, (uint32_t)type->outputs.size()) : 1;
        for (uint32_t output_id = 0; output_id < num_outputs; ++output_id) {
            ResourceUUID rid{vn.id, output_id};
            if (active_resources && !active_resources->count(rid)) continue;
            total += (size_t)width * height
                   * storage_format_bytes(resolve_node_storage(vn, lib, output_id));
        }
    }
    if (total > budget_bytes_) {
        std::string e = "Memory Budget Exceeded: Need "
                        + std::to_string(total / (1024 * 1024)) + " MB, budget "
                        + std::to_string(budget_bytes_ / (1024 * 1024)) + " MB";
        if (error) *error = e;
        log_error(e);
        return false;
    }

    // Build the allocation list (only nodes whose type is in the library).
    struct ResInfo {
        ResourceUUID rid;
        uint32_t color = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        std::string debug_name;
    };
    std::vector<ResInfo> all_res;
    std::unordered_map<uint32_t, std::vector<size_t>> color_to_indices;
    for (const auto& vn : ir.nodes) {
        const auto* type = lib.find(vn.type_id);
        if (!type) continue;
        const uint32_t num_outputs = std::max(1u, (uint32_t)type->outputs.size());
        for (uint32_t output_id = 0; output_id < num_outputs; ++output_id) {
            ResourceUUID rid{vn.id, output_id};
            if (active_resources && !active_resources->count(rid)) continue;
            const StorageFormat sf = resolve_node_storage(vn, lib, output_id);
            const VkFormat fmt = storage_format_to_vk(sf);
            uint32_t cc = 0;
            if (color_classes) {
                auto cit = color_classes->find(rid);
                if (cit != color_classes->end()) cc = cit->second;
            }
            size_t idx = all_res.size();
            all_res.push_back({rid, cc, fmt, vn.debug_name + "_out" + std::to_string(output_id)});
            color_to_indices[cc].push_back(idx);
        }
    }

    // -- Stage B: allocate pinned / non-aliased resources ---------------
    // A resource is "non-aliased" iff its color class is 0 (pinned) OR
    // its class has only 1 member (a singleton that doesn't share memory).
    // Such resources go through the normal vmaCreateImage path with
    // alias_group_id = 0. Resources in groups of 2+ are handled in Stage C.
    for (size_t i = 0; i < all_res.size(); ++i) {
        const auto& info = all_res[i];
        bool in_alias_group = false;
        if (info.color != 0) {
            auto it = color_to_indices.find(info.color);
            if (it != color_to_indices.end() && it->second.size() >= 2) {
                in_alias_group = true;
            }
        }
        if (in_alias_group) continue;

        NodeResource r;
        r.node_id = info.rid.node_id;
        r.output_index = info.rid.output_index;
        r.format = info.format;
        r.alias_group_id = 0;
        if (!create_image_(ctx, r, width, height, info.debug_name)) {
            if (error) *error = "image allocation failed for " + info.debug_name;
            return false;
        }
        log_info("[mem]   created image: node=" + std::to_string(info.rid.node_id)
                 + " output=" + std::to_string(info.rid.output_index)
                 + " " + std::to_string(width) + "x" + std::to_string(height)
                 + " fmt=" + std::to_string(info.format)
                 + " " + std::to_string((size_t)width * height * pixel_bytes_(info.format) / 1024) + " KB"
                 + " [pinned]");
        live_[info.rid] = std::move(r);
        current_bytes_ += (size_t)width * height * pixel_bytes_(info.format);
    }

    // -- Stage C: allocate aliased color classes (manual bind) -----------
    //
    // VMA doc pattern (resource_aliasing.html): for each group of images
    // with non-overlapping lifetimes, allocate ONE VmaAllocation sized to
    // max(member) and bind every image in the group to it via
    // vmaBindImageMemory. This is the only path that guarantees physical
    // memory overlap in VMA -- the pool+CAN_ALIAS_BIT path does NOT.
    for (auto& kv : color_to_indices) {
        uint32_t color = kv.first;
        const auto& indices = kv.second;
        if (color == 0 || indices.size() < 2) continue;

        const ResInfo& first_info = all_res[indices[0]];
        size_t N = indices.size();

        // C.1: create every VkImage up front (no VMA binding yet).
        std::vector<VkImage>              images(N, VK_NULL_HANDLE);
        std::vector<VkMemoryRequirements> reqs(N);
        std::vector<NodeResource>         resources(N);
        bool any_failed = false;
        for (size_t k = 0; k < N && !any_failed; ++k) {
            const auto& info = all_res[indices[k]];
            NodeResource& r = resources[k];
            r.node_id      = info.rid.node_id;
            r.output_index = info.rid.output_index;
            r.format       = info.format;
            r.debug_name   = info.debug_name;
            r.extent       = {width, height, 1};
            r.layout       = VK_IMAGE_LAYOUT_UNDEFINED;

            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType     = VK_IMAGE_TYPE_2D;
            ici.format        = r.format;
            ici.extent        = r.extent;
            ici.mipLevels     = 1;
            ici.arrayLayers   = 1;
            ici.samples       = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
            ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT
                              | VK_IMAGE_USAGE_SAMPLED_BIT
                              | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                              | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            ici.flags        |= VK_IMAGE_CREATE_ALIAS_BIT;

            std::string usage_error;
            if (!ctx.format_supports_image_usage(r.format, ici.usage, &usage_error)) {
                log_error("Aliasing: " + usage_error + " for " + info.debug_name);
                if (error) *error = usage_error;
                any_failed = true;
                break;
            }

            if (vkCreateImage(ctx.device(), &ici, nullptr, &images[k]) != VK_SUCCESS) {
                log_error("Aliasing: vkCreateImage failed for " + info.debug_name);
                if (error) *error = "vkCreateImage failed for " + info.debug_name;
                any_failed = true;
                break;
            }
            ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE, (uint64_t)images[k], info.debug_name);
            vkGetImageMemoryRequirements(ctx.device(), images[k], &reqs[k]);
        }
        if (any_failed) {
            for (auto img : images) if (img) vkDestroyImage(ctx.device(), img, nullptr);
            return false;
        }

        // C.2: combine memory requirements. size = max, alignment = max,
        // memoryTypeBits = AND across all members.
        VkMemoryRequirements combined{};
        combined.size           = reqs[0].size;
        combined.alignment      = reqs[0].alignment;
        combined.memoryTypeBits = reqs[0].memoryTypeBits;
        for (size_t k = 1; k < N; ++k) {
            combined.size           = std::max(combined.size, reqs[k].size);
            combined.alignment      = std::max(combined.alignment, reqs[k].alignment);
            combined.memoryTypeBits &= reqs[k].memoryTypeBits;
        }
        if (combined.memoryTypeBits == 0) {
            log_error("Aliasing: combined memoryTypeBits==0 for color " + std::to_string(color)
                      + " (resources have disjoint memory-type requirements)");
            for (auto img : images) if (img) vkDestroyImage(ctx.device(), img, nullptr);
            if (error) *error = "aliasing: disjoint memoryTypeBits";
            return false;
        }
        log_info("Aliasing: color=" + std::to_string(color)
                 + " N=" + std::to_string(N)
                 + " size=" + std::to_string(combined.size)
                 + " alignment=" + std::to_string(combined.alignment)
                 + " memTypeBits=" + std::to_string(combined.memoryTypeBits));

        // C.3: one VmaAllocation for the whole group. Matches the VMA
        // doc's verbatim aliasing example (resource_aliasing.html):
        // preferredFlags = DEVICE_LOCAL, no CAN_ALIAS_BIT (the latter
        // prevents VMA from attaching VkMemoryDedicatedAllocateInfoKHR
        // and we don't need that hint here -- the aliasing semantics
        // come from binding multiple VkImages to the same allocation).
        VmaAllocationCreateInfo aci{};
        aci.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        VmaAllocation shared_alloc = nullptr;
        VkResult r_alloc = vmaAllocateMemory(ctx.allocator(), &combined, &aci, &shared_alloc, nullptr);
        if (r_alloc != VK_SUCCESS) {
            log_error("Aliasing: vmaAllocateMemory failed for color " + std::to_string(color)
                      + " (VkResult=" + std::to_string(r_alloc)
                      + " size=" + std::to_string(combined.size) + ")");
            for (auto img : images) if (img) vkDestroyImage(ctx.device(), img, nullptr);
            if (error) *error = "vmaAllocateMemory failed";
            return false;
        }

        // C.4: bind every image to the shared allocation.
        bool bind_failed = false;
        for (size_t k = 0; k < N; ++k) {
            if (vmaBindImageMemory(ctx.allocator(), shared_alloc, images[k]) != VK_SUCCESS) {
                log_error("Aliasing: vmaBindImageMemory failed for " + all_res[indices[k]].debug_name);
                if (error) *error = "vmaBindImageMemory failed for " + all_res[indices[k]].debug_name;
                bind_failed = true;
                break;
            }
        }
        if (bind_failed) {
            vmaFreeMemory(ctx.allocator(), shared_alloc);
            for (auto img : images) if (img) vkDestroyImage(ctx.device(), img, nullptr);
            return false;
        }

        // C.5: register the alias group. The PRIMARY is the first member
        // (indices[0]); it owns the shared VmaAllocation. Siblings have
        // alloc = nullptr and rely on the primary's destruction to free
        // the underlying VkDeviceMemory.
        uint32_t group_id = (uint32_t)(alias_pools_.size() + 1);
        alias_pools_.push_back({color, first_info.format, {width, height, 1},
                                first_info.rid, 1});
        vmaSetAllocationName(ctx.allocator(), shared_alloc,
                             (first_info.debug_name + "_alias").c_str());

        for (size_t k = 0; k < N; ++k) {
            NodeResource& r = resources[k];
            r.image = images[k];
            r.alloc = (k == 0) ? shared_alloc : nullptr;  // primary owns alloc
            r.alias_group_id = group_id;

            VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            vci.image    = r.image;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format   = r.format;
            vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(ctx.device(), &vci, nullptr, &r.view) != VK_SUCCESS) {
                log_error("Aliasing: view creation failed for " + r.debug_name);
                if (error) *error = "view creation failed for " + r.debug_name;
                vmaFreeMemory(ctx.allocator(), shared_alloc);
                for (auto img : images) if (img) vkDestroyImage(ctx.device(), img, nullptr);
                return false;
            }
            ctx.set_debug_name(VK_OBJECT_TYPE_IMAGE_VIEW, (uint64_t)r.view,
                               r.debug_name + "_view");

            const auto& info = all_res[indices[k]];
            log_info("[mem]   created image: node=" + std::to_string(info.rid.node_id)
                     + " output=" + std::to_string(info.rid.output_index)
                     + " " + std::to_string(width) + "x" + std::to_string(height)
                     + " fmt=" + std::to_string(info.format)
                     + " " + std::to_string((size_t)width * height * pixel_bytes_(info.format) / 1024) + " KB"
                     + " [alias group=" + std::to_string(group_id)
                     + (k == 0 ? " primary" : " sibling") + "]");
            live_[info.rid] = std::move(r);
            current_bytes_ += (size_t)width * height * pixel_bytes_(info.format);
        }
    }

    log_info("ResourceManager: Allocated " + std::to_string(live_.size())
             + " images, " + std::to_string(current_bytes_ / (1024 * 1024)) + " MB logical"
             + (alias_pools_.empty() ? "" : ", " + std::to_string(alias_pools_.size()) + " alias groups"));
    return true;
}


bool ResourceManager::allocate_for_preview(VulkanContext& ctx,
                                           const GraphIR& ir,
                                           const NodeLibrary& lib,
                                           ResourceUUID output_rid,
                                           uint32_t width, uint32_t height,
                                           VkFormat default_format) {
    retire_all(ctx);
    alias_pools_.clear();

    const auto* vn = ir.find(output_rid.node_id);
    const auto* type = vn ? lib.find(vn->type_id) : nullptr;
    VkFormat fmt = default_format;
    if (vn) {
        const StorageFormat sf = resolve_node_storage(*vn, lib, output_rid.output_index);
        fmt = storage_format_to_vk(sf);
    }

    NodeResource r;
    r.node_id = output_rid.node_id;
    r.output_index = output_rid.output_index;
    r.format = fmt;
    r.alias_group_id = 0;
    std::string dbg = (vn ? vn->debug_name : "output") + "_out" + std::to_string(output_rid.output_index);
    if (!create_image_(ctx, r, width, height, dbg)) {
        log_error("ResourceManager: preview image allocation failed for " + dbg);
        return false;
    }
    live_[output_rid] = std::move(r);
    current_bytes_ = (size_t)width * height * pixel_bytes_(fmt);

    log_info("ResourceManager: Preview allocated 1 image, "
             + std::to_string(current_bytes_ / (1024 * 1024)) + " MB");
    return true;
}


void ResourceManager::retire_all(VulkanContext& ctx) {
    for (auto& kv : live_) {
        retired_.push_back({std::move(kv.second), RETIRE_FRAMES});
    }
    live_.clear();
    current_bytes_ = 0;
    (void)ctx;
}


void ResourceManager::tick(VulkanContext& ctx) {
    for (auto& r : retired_) if (r.frames_remaining > 0) --r.frames_remaining;
    // Two passes: siblings (alloc==nullptr) must be destroyed before the
    // primary that owns their shared VmaAllocation. Otherwise the
    // primary's vmaDestroyImage would free memory still referenced by a
    // sibling's VkImage, triggering a validation error.
    size_t siblings_destroyed = 0;
    size_t primaries_destroyed = 0;
    retired_.erase(
        std::remove_if(retired_.begin(), retired_.end(),
            [&](Retired& r) {
                if (r.frames_remaining == 0 && r.res.alloc == nullptr) {
                    destroy_image_(ctx, r.res);
                    ++siblings_destroyed;
                    return true;
                }
                return false;
            }),
        retired_.end());
    retired_.erase(
        std::remove_if(retired_.begin(), retired_.end(),
            [&](Retired& r) {
                if (r.frames_remaining == 0) {
                    destroy_image_(ctx, r.res);
                    ++primaries_destroyed;
                    return true;
                }
                return false;
            }),
        retired_.end());
    if (siblings_destroyed > 0 || primaries_destroyed > 0) {
        log_info("[mem] tick retired destroy: siblings=" + std::to_string(siblings_destroyed)
                 + " primaries=" + std::to_string(primaries_destroyed)
                 + " remaining_retired=" + std::to_string(retired_.size()));
    }
}

void ResourceManager::shutdown(VulkanContext& ctx) {
    for (auto& kv : live_)    if (kv.second.alloc == nullptr) destroy_image_(ctx, kv.second);
    for (auto& r  : retired_) if (r.res.alloc      == nullptr) destroy_image_(ctx, r.res);
    for (auto& kv : live_)    if (kv.second.alloc != nullptr) destroy_image_(ctx, kv.second);
    for (auto& r  : retired_) if (r.res.alloc      != nullptr) destroy_image_(ctx, r.res);
    alias_pools_.clear();
    live_.clear();
    retired_.clear();
    current_bytes_ = 0;
}

VmaStatsReport ResourceManager::get_vma_stats(VulkanContext& ctx) const {
    VmaStatsReport r;
    r.node_resource_count      = live_.size();
    r.node_resource_bytes      = current_bytes_;
    r.warning_threshold_bytes  = budget_bytes_;
    r.retired_count            = retired_.size();
    for (const auto& ret : retired_) {
        // Re-derive from extents rather than tracking per-retire bytes;
        // the cost is trivial (retired list is bounded by MAX_FRAMES_IN_FLIGHT+1)
        // and avoids a separate accumulator.
        r.retired_bytes += (size_t)ret.res.extent.width
                         * (size_t)ret.res.extent.height
                         * pixel_bytes_(ret.res.format);
    }

    // VMA API: vmaCalculateStatistics fills VmaTotalStatistics; total is rolled-up, heapStats is per-heap.
    VmaTotalStatistics ts{};
    vmaCalculateStatistics(ctx.allocator(), &ts);
    const VmaStatistics& s = ts.total.statistics;
    r.vma_block_bytes         = (size_t)s.blockBytes;
    r.vma_allocation_bytes    = (size_t)s.allocationBytes;
    r.vma_unused_range_bytes  = (size_t)(s.blockBytes - s.allocationBytes);

    // Per-heap budget / usage. Real OS-level numbers with EXT_MEMORY_BUDGET_BIT; 80% heuristic fallback otherwise. Edge cases: memoryHeapCount==0, zero budget, usage>budget, UMA systems.
    const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
    vmaGetMemoryProperties(ctx.allocator(), &mem_props);
    if (mem_props && mem_props->memoryHeapCount > 0) {
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
        vmaGetHeapBudgets(ctx.allocator(), budgets);

        r.heap_stats.reserve(mem_props->memoryHeapCount);
        for (uint32_t i = 0; i < mem_props->memoryHeapCount; ++i) {
            const VkMemoryHeap& heap = mem_props->memoryHeaps[i];
            const VmaBudget&    b    = budgets[i];

            VmaHeapStats h;
            h.index                = i;
            h.is_device_local      = (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            h.label                = h.is_device_local ? "DEVICE_LOCAL" : "HOST";
            h.heap_size_bytes      = (size_t)heap.size;
            h.budget_bytes         = (size_t)b.budget;
            h.usage_bytes          = (size_t)b.usage;
            h.vma_block_bytes      = (size_t)b.statistics.blockBytes;
            h.vma_allocation_bytes = (size_t)b.statistics.allocationBytes;
            h.vma_block_count      = b.statistics.blockCount;
            h.vma_allocation_count = b.statistics.allocationCount;

            // Pressure: usage / budget, clamped 0..1. Zero when budget is 0 (avoids div-by-zero).
            if (h.budget_bytes > 0) {
                h.pressure = (float)double(h.usage_bytes) / (float)double(h.budget_bytes);
                if (h.pressure < 0.0f) h.pressure = 0.0f;
                if (h.pressure > 1.0f) h.pressure = 1.0f;
            }

            if (h.is_device_local) {
                r.gpu_budget_bytes += h.budget_bytes;
                r.gpu_usage_bytes  += h.usage_bytes;
                r.device_local_allocation_bytes += h.vma_allocation_bytes;
            }
            r.heap_stats.push_back(h);
        }
        if (r.gpu_budget_bytes > 0) {
            r.gpu_pressure = (float)double(r.gpu_usage_bytes) / (float)double(r.gpu_budget_bytes);
            if (r.gpu_pressure < 0.0f) r.gpu_pressure = 0.0f;
            if (r.gpu_pressure > 1.0f) r.gpu_pressure = 1.0f;
        }
    }
    return r;
}


} // namespace te
