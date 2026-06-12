#pragma once

#include "engine/VulkanContext.hpp"
#include "engine/ShaderCompiler.hpp"
#include "engine/ShaderCache.hpp"
#include "engine/ComputePipeline.hpp"
#include "engine/Image.hpp"
#include "engine/ImageUploader.hpp"
#include "engine/PushConstants.hpp"
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/ResourceManager.hpp"
#include "engine/BindlessTable.hpp"
#include "engine/PassPlan.hpp"
#include "engine/AsyncReadback.hpp"
#include "engine/DirtySet.hpp"
#include "engine/EngineError.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace te {

constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

#define TE_GUARD_READY(call)                                                     \
    std::lock_guard<std::recursive_mutex> te_lk(entry_mu_);                       \
    if (state_.load(std::memory_order_acquire) != EngineState::Ready) {          \
        set_error_(EngineErrorCode::UseAfterShutdown,                            \
                   std::string("engine not Ready (state=")                       \
                   + std::to_string((int)state_.load()) + ")",                   \
                   EnginePhase::Idle);                                           \
        return call;                                                             \
    }

inline const VkSpecializationInfo* build_spec_info(const ShaderVariantKey& vk,
                                                   VkSpecializationMapEntry (&out_entries)[8],
                                                   VkSpecializationInfo&   out_info) {
    if (vk.specialization_count == 0) return nullptr;
    const uint32_t n = vk.specialization_count > 8 ? 8 : vk.specialization_count;
    for (uint32_t i = 0; i < n; ++i) {
        out_entries[i].constantID = i;
        out_entries[i].offset     = i * sizeof(uint32_t);
        out_entries[i].size       = sizeof(uint32_t);
    }
    out_info.mapEntryCount = n;
    out_info.pMapEntries   = out_entries;
    out_info.dataSize      = n * sizeof(uint32_t);
    out_info.pData         = vk.specialization.data();
    return &out_info;
}

struct PassExec {
    std::unique_ptr<ComputePipeline> pipeline;
    NodeId                           node_id = 0;
    std::vector<ResourceUUID>        input_resources;
    std::vector<ChannelFormat>       input_formats;
    std::vector<ResourceUUID>        output_resources;
    int                              param_base_slot = 0;

    uint32_t out_storage_slots[MAX_PASS_OUTPUTS] = {};
    uint32_t output_count = 0;
    uint32_t in_sampled_slots[MAX_PASS_INPUTS] = {};
    uint32_t input_count = 0;

    bool bypassed = false;
};

struct RetiredPass {
    std::unique_ptr<ComputePipeline> pipeline;
    uint32_t frames_remaining = MAX_FRAMES_IN_FLIGHT + 2;
};

struct PassTiming {
    uint32_t     pass_index   = 0;
    double       duration_us  = 0.0;
    bool         available    = false;
};

struct TimestampPool {
    VkQueryPool           pool         = VK_NULL_HANDLE;
    uint32_t              capacity     = 0;
    std::vector<PassTiming> cached_timings;
};

struct ChainExec {
    std::unique_ptr<ComputePipeline> pipeline;
    std::vector<uint32_t>            member_pass_indices;
    uint32_t                         head_pass_index = 0;
    uint32_t                         tail_pass_index = 0;
    bool                             bypassed        = false;
    uint32_t                         chain_in_sampled_slots[MAX_PASS_INPUTS] = {};
};

class Engine {
public:
    bool init(VkSurfaceKHR surface,
              const char** extra_inst_exts, uint32_t extra_inst_ext_count,
              bool enable_validation,
              const std::string& cache_dir,
              const std::string& nodes_dir,
              const std::string& glsl_dir);
    void shutdown();

    uint64_t set_graph(const Graph& graph);

    uint64_t set_active_node(NodeId node_id);

    struct BakedImage {
        std::string name;
        uint32_t    width  = 0;
        uint32_t    height = 0;
        std::vector<float> pixels;
    };
    std::vector<BakedImage> bake();
    BakedImage readback_sync();

    const NodeLibrary& node_library() const { return node_lib_; }
    const Graph& current_graph() const { return current_graph_; }

    void poll_pending_compiles();
    void tick_retired();

    void update_node_params_by_id(NodeId node_id, const std::vector<float>& params);
    void update_node_params_by_name(NodeId node_id,
                                    const std::unordered_map<std::string, float>& kv);
    const std::unordered_map<NodeId,int>& param_layout() const { return param_base_slot_; }
    int total_param_floats() const { return total_param_floats_; }

    void set_precision(int mode) {
        if (mode == 0)      texture_format_ = VK_FORMAT_R8G8B8A8_UNORM;
        else if (mode == 1) texture_format_ = VK_FORMAT_R16G16B16A16_SFLOAT;
        else                texture_format_ = VK_FORMAT_R32G32B32A32_SFLOAT;
    }
    int precision() const {
        if (texture_format_ == VK_FORMAT_R8G8B8A8_UNORM)      return 0;
        if (texture_format_ == VK_FORMAT_R16G16B16A16_SFLOAT) return 1;
        return 2;
    }
    VkFormat texture_format() const { return texture_format_; }

    void set_resolution(uint32_t w, uint32_t h) {
        if (output_w_ == w && output_h_ == h) return;
        output_w_ = w;
        output_h_ = h;
        if (!ctx_.device()) return;
        if (output_storage_ && output_storage_->image() != VK_NULL_HANDLE) {
            RetiredImage ri;
            ri.img = std::move(output_storage_);
            ri.frames_remaining = MAX_FRAMES_IN_FLIGHT + 2;
            retired_images_.push_back(std::move(ri));
        }
        output_storage_ = std::make_unique<Image>();
        output_storage_->create(ctx_, output_w_, output_h_);
        output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        dirty_set_.mark_topology_change();
        any_pass_dirty_.store(true);
        last_presented_pixels_.clear();
        last_presented_w_ = 0;
        last_presented_h_ = 0;
    }

    void record_dispatch(VkCommandBuffer cmd, const PushConstants& pc);

    bool has_pipeline() const { return !passes_.empty(); }
    uint64_t last_dispatch_count() const noexcept { return last_dispatch_count_; }
    uint64_t compile_generation()  const { return compile_generation_; }
    uint64_t installed_generation() const { return installed_generation_; }
    bool is_generation_ready(uint64_t generation) const {
        return generation != 0 && has_pipeline() && installed_generation_ == generation;
    }
    const std::string& last_error() const { return last_error_; }
    NodeId failed_node() const { return failed_node_id_; }

    const EngineError& last_error_record() const { return last_error_record_; }
    void set_error_record(EngineError e) {
        last_error_record_ = std::move(e);
        last_error_     = last_error_record_.message;
        failed_node_id_ = last_error_record_.failed_node;
    }
    void clear_error() {
        last_error_record_.clear();
        last_error_.clear();
        failed_node_id_ = 0;
    }

    enum class EngineState : uint8_t {
        Uninitialized = 0,
        Initializing  = 1,
        Ready         = 2,
        ShuttingDown  = 3,
        ShutDown      = 4,
        Error         = 5,
    };
    EngineState engine_state() const noexcept { return state_.load(std::memory_order_acquire); }
    bool is_ready() const noexcept { return engine_state() == EngineState::Ready; }

    std::recursive_mutex& entry_mutex() noexcept { return entry_mu_; }

    uint64_t last_dispatch_count_ = 0;

    GraphRevisionId current_revision() const { return current_revision_; }
    const GraphIR&  current_ir()       const { return current_ir_; }

    VulkanContext&   ctx()       { return ctx_; }
    Image&           output()    { return *output_storage_; }
    ResourceManager& resources() { return resources_; }
    size_t           resources_live_count() const { return resources_.live_count(); }

    AsyncReadback& async_readback() { return async_; }

    const std::vector<PassTiming>& last_pass_timings() const { return last_pass_timings_; }

    bool any_pass_dirty() const noexcept       { return any_pass_dirty_.load(); }
    void mark_all_clean() { 
        any_pass_dirty_.store(false);
        dirty_set_.clear();
    }
    void mark_node_dirty(NodeId node_id);
    bool has_presented_frame() const noexcept  { return last_presented_w_ > 0; }
    uint64_t republish_last_frame(uint64_t generation);
    void stash_last_presented(const std::vector<float>& pixels,
                              uint32_t w, uint32_t h, uint64_t generation);

    ResourceUUID final_output_resource() const { return final_output_resource_; }

    bool upload_image(uint64_t node_id, const float* pixels, uint32_t width, uint32_t height);
    bool release_image(uint64_t node_id);


private:
    void retire_all_passes_();
    bool ensure_dummy_images_();
    bool create_global_samplers_();
    void destroy_global_samplers_();
    void mark_downstream_dirty_(NodeId root);
    void rebuild_downstream_adj_();

    void populate_chains_(const PassPlan& plan);
    void record_chain_dispatch_(VkCommandBuffer cmd, const PushConstants& pc,
                                uint32_t gx, uint32_t gy, size_t chain_idx);
    void record_barriers_(VkCommandBuffer cmd, const PushConstants& pc);
    bool record_pass_dispatches_(VkCommandBuffer cmd, const PushConstants& pc,
                                 uint32_t gx, uint32_t gy);
    void record_final_blit_(VkCommandBuffer cmd, const PushConstants& pc,
                            bool final_pass_was_dirty);
    void seed_param_ssbo_defaults_();
    void poll_completed_uploads_();

    static constexpr uint32_t TIMESTAMP_POOL_COUNT = 3;
    bool create_timestamp_pools_();
    void destroy_timestamp_pools_();
    void poll_timestamps_();
    uint32_t ts_pool_write_idx_ = 0;
    uint32_t ts_query_idx_ = 0;
    std::array<TimestampPool, TIMESTAMP_POOL_COUNT> ts_pools_{};
    std::vector<PassTiming> last_pass_timings_;

    void shutdown_internal_();
    void assign_bindless_slots_(PassExec& pe);
    bool resolve_aliased_staleness_();

    bool create_pass_pipeline_(PassExec& pe,
                               NodeId node_id,
                               const std::string& type_id,
                               const std::string& error_prefix,
                               const ShaderVariantKey& variant_key,
                               const std::vector<uint32_t>& spirv);

    void set_error_(EngineErrorCode code,
                    std::string message,
                    EnginePhase phase,
                    NodeId failed_node = 0,
                    uint64_t graph_generation = 0);

    VulkanContext   ctx_;
    ShaderCompiler  compiler_;
    std::unique_ptr<ShaderCache> cache_;
    NodeLibrary     node_lib_;
    Graph           current_graph_;
    GraphIR         current_ir_;
    GraphRevisionId current_revision_ = 0;

    ImageUploader uploader_;
    struct PendingUpload { uint64_t node_id = 0; uint64_t ticket  = 0; };
    std::vector<PendingUpload> pending_uploads_;
    std::unordered_map<uint64_t, bool> images_needing_acquire_;

    std::array<Image, 6>    dummy_images_;
    std::array<uint32_t, 6> dummy_sampled_slots_{
        BindlessTable::INVALID_SLOT, BindlessTable::INVALID_SLOT,
        BindlessTable::INVALID_SLOT, BindlessTable::INVALID_SLOT,
        BindlessTable::INVALID_SLOT, BindlessTable::INVALID_SLOT
    };

    ResourceManager resources_;

    uint32_t output_w_ = 512;
    uint32_t output_h_ = 512;

    uint64_t compile_generation_   = 0;
    uint64_t pending_generation_   = 0;
    uint64_t installed_generation_ = 0;

    static constexpr uint32_t PARAM_RING = MAX_FRAMES_IN_FLIGHT + 1;
    static_assert(PARAM_RING == BindlessTable::PARAM_RING_SIZE,
                  "Engine PARAM_RING must match BindlessTable::PARAM_RING_SIZE");
    VkBuffer       param_buf_[PARAM_RING]    {};
    VmaAllocation  param_alloc_[PARAM_RING]  {};
    void*          param_mapped_[PARAM_RING] {};
    bool           param_dirty_ = false;
    uint32_t       param_write_idx_ = 0;

    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> res_sampled_slot_;
    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> res_storage_slot_;
    std::unordered_map<uint64_t, uint32_t> ext_sampled_slot_;

    BindlessTable  bindless_;

    std::unique_ptr<Image> output_storage_;
    struct RetiredImage { std::unique_ptr<Image> img; uint32_t frames_remaining; };
    std::vector<RetiredImage> retired_images_;

    std::vector<PassExec> passes_;
    std::vector<RetiredPass> retired_passes_;
    ResourceUUID  final_output_resource_ = {};

    std::vector<ChainExec> chain_execs_;
    std::vector<uint32_t>   chain_id_of_pass_;

    std::unordered_map<uint64_t, std::unique_ptr<Image>> image_registry_;
    VkImageLayout dummy_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    std::unordered_map<NodeId, std::vector<NodeId>> downstream_adj_;

    struct PendingPass {
        std::future<CompileResult> fut;
        std::string                name;
        std::string                type_id;
        uint32_t                   input_count = 0;
        std::vector<ResourceUUID>  output_resources;
        std::vector<ResourceUUID>  input_resources;
        std::vector<ChannelFormat> input_formats;
        NodeId                     node_id = 0;
        PassKind                   kind = PassKind::Compute;
        ShaderVariantKey           variant_key;
        bool                       bypassed = false;
        bool                       chain_member = false;
        int                        param_base_slot = 0;
    };

    std::vector<PendingPass> pending_passes_;
    bool                     pending_active_ = false;
    ResourceUUID             pending_final_output_ = {};
    PassPlan                 pending_pass_plan_;

    std::unordered_map<NodeId, int> param_base_slot_;
    int total_param_floats_ = 0;
    std::string last_error_;
    NodeId failed_node_id_ = 0;
    EngineError last_error_record_;

    VkSampler sampler_repeat_ = VK_NULL_HANDLE;
    VkSampler sampler_clamp_  = VK_NULL_HANDLE;
    VkSampler sampler_mirror_ = VK_NULL_HANDLE;
    VkFormat  texture_format_ = VK_FORMAT_R32G32B32A32_SFLOAT;

    std::atomic<bool> any_pass_dirty_{true};

    std::atomic<EngineState> state_{EngineState::Uninitialized};
    std::recursive_mutex     entry_mu_;
    std::vector<float> last_presented_pixels_;
    uint32_t           last_presented_w_{0};
    uint32_t           last_presented_h_{0};

    DirtySet          dirty_set_;

    AsyncReadback     async_;
};

} // namespace te
