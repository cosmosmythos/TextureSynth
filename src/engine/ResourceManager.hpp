#pragma once
#include "engine/NodeResource.hpp"
#include "engine/GraphIR.hpp"
#include <unordered_map>
#include <vector>
#include <cstddef>

namespace te {

class VulkanContext;

class ResourceManager {
public:
    // Soft limit, MB. Default 1 GB. Configurable later.
    void set_memory_budget_mb(size_t mb) { budget_bytes_ = mb * 1024ull * 1024ull; }

    // Allocate one image per (node_id, output_index=0) in the active subgraph.
    // Returns false if budget would be exceeded; populates `error`.
    bool allocate_for_graph(VulkanContext& ctx,
                            const GraphIR& ir,
                            uint32_t width, uint32_t height,
                            VkFormat format,
                            std::string* error = nullptr);

    // Defer destruction; caller must call tick() each frame.
    void retire_all(VulkanContext& ctx);
    void tick(VulkanContext& ctx);
    void shutdown(VulkanContext& ctx); // destroys everything immediately

    const NodeResource* get(ResourceUUID id) const {
        auto it = live_.find(id);
        return it == live_.end() ? nullptr : &it->second;
    }
    NodeResource* get(ResourceUUID id) {
        auto it = live_.find(id);
        return it == live_.end() ? nullptr : &it->second;
    }

    size_t current_bytes() const { return current_bytes_; }
    size_t budget_bytes() const  { return budget_bytes_; }
    size_t live_count() const { return live_.size(); }

private:
    struct Retired {
        NodeResource res;
        uint32_t frames_remaining;
    };

    size_t pixel_bytes_(VkFormat f) const; // returns bytes per pixel
    bool create_image_(VulkanContext& ctx, NodeResource& r,
                       uint32_t w, uint32_t h, const std::string& dbg);
    void destroy_image_(VulkanContext& ctx, NodeResource& r);

    std::unordered_map<ResourceUUID, NodeResource, ResourceUUIDHash> live_;
    std::vector<Retired> retired_;
    size_t current_bytes_ = 0;
    size_t budget_bytes_  = 1024ull * 1024ull * 1024ull; // 1 GB default
};

} // namespace te