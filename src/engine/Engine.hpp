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
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>

namespace te {

constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

struct PassExec {
    std::unique_ptr<ComputePipeline> pipeline;
    NodeId                           node_id = 0;
    std::vector<ResourceUUID>        input_resources;
    std::vector<ResourceUUID>        output_resources;
    int                              param_base_slot = 0;

    // Bindless: precomputed push-constant payload (slots only).
    uint32_t out_storage_slots[MAX_PASS_OUTPUTS] = {};
	uint32_t output_count = 0;
    uint32_t in_sampled_slots[MAX_PASS_INPUTS] = {};
    uint32_t input_count = 0;

    bool output_layout_is_general = false;

    // Phase 1c: mirror of ComputePass::bypassed. When true, the executor
    // skips the compute dispatch and issues vkCmdClearColorImage on each
    // output resource (writes zero, leaves image in VK_IMAGE_LAYOUT_GENERAL).
    // Bypassed passes still allocate output storage / sampled bindless
    // slots (assign_bindless_slots_ runs) so downstream passes that read
    // the bypassed output get a valid (zeroed) source.
    bool bypassed = false;
};

struct RetiredPass {
    std::unique_ptr<ComputePipeline> pipeline;
    uint32_t frames_remaining = MAX_FRAMES_IN_FLIGHT + 2;
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

        // Retire current output_ — don't destroy until GPU is done with it.
        // The async ring drains naturally as fences signal; we just defer.
        if (output_storage_ && output_storage_->image() != VK_NULL_HANDLE) {
            RetiredImage ri;
            ri.img = std::move(output_storage_);  // see below
            ri.frames_remaining = MAX_FRAMES_IN_FLIGHT + 2;
            retired_images_.push_back(std::move(ri));
        }
        output_storage_ = std::make_unique<Image>();
        output_storage_->create(ctx_, output_w_, output_h_);
        output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

        // Force a full re-render at the new resolution.
        dirty_set_.mark_topology_change();
        any_pass_dirty_.store(true);
        last_presented_pixels_.clear();
        last_presented_w_ = 0;
        last_presented_h_ = 0;
    }

    // Phase 5: records the full PassPlan with per-pass barriers.
    void record_dispatch(VkCommandBuffer cmd, const PushConstants& pc);

    bool has_pipeline() const { return !passes_.empty(); }
    uint64_t compile_generation()  const { return compile_generation_; }
    uint64_t installed_generation() const { return installed_generation_; }
    bool is_generation_ready(uint64_t generation) const {
        return generation != 0 && has_pipeline() && installed_generation_ == generation;
    }
    const std::string& last_error() const { return last_error_; }
    NodeId failed_node() const { return failed_node_id_; }

    // Phase 0: structured error channel. Survives until the next successful op.
    const EngineError& last_error_record() const { return last_error_record_; }
    void set_error_record(EngineError e) {
        last_error_record_ = std::move(e);
        // Mirror to legacy fields for callers that still read the old API.
        last_error_     = last_error_record_.message;
        failed_node_id_ = last_error_record_.failed_node;
    }
    void clear_error() {
        last_error_record_.clear();
        last_error_.clear();
        failed_node_id_ = 0;
    }

    // Phase 5: explicit lifecycle state machine. Every public mutator guards
    // on this so use-after-shutdown is a clean EngineError, not a crash.
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

    // The mutex that serialises every public mutator. Exposed for the
    // bindings layer (and for tests) so they can hold the lock for the
    // full duration of a multi-step operation, eliminating the window
    // where Engine::shutdown could interleave with an in-flight call.
    std::mutex& entry_mutex() noexcept { return entry_mu_; }

    GraphRevisionId current_revision() const { return current_revision_; }
    const GraphIR&  current_ir()       const { return current_ir_; }

    VulkanContext&   ctx()       { return ctx_; }
    Image&           output()    { return *output_storage_; }
    ResourceManager& resources() { return resources_; }
    size_t           resources_live_count() const { return resources_.live_count(); }

    // Async readback ring. Public because bindings.cpp drives it.
    AsyncReadback& async_readback() { return async_; }

    // ── Dirty tracking: skip GPU submit when nothing changed ──────────
    bool any_pass_dirty() const noexcept       { return any_pass_dirty_.load(); }
    void mark_all_clean() noexcept             { any_pass_dirty_.store(false); }
    void mark_node_dirty(NodeId node_id);      // sets flag + propagates downstream
    bool has_presented_frame() const noexcept  { return last_presented_w_ > 0; }
    uint64_t republish_last_frame(uint64_t generation);
    void stash_last_presented(const std::vector<float>& pixels,
                              uint32_t w, uint32_t h, uint64_t generation);

    // Resource UUID written by the final pass — readback copies from this image.
    ResourceUUID final_output_resource() const { return final_output_resource_; }

    // Persistent external images uploaded from Blender (keyed by node stable ID)
    bool upload_image(uint64_t node_id, const float* pixels, uint32_t width, uint32_t height);
    bool release_image(uint64_t node_id);


private:
    // bool build_passes_from_plan_(const PassPlan& plan);
    void retire_all_passes_();
    bool ensure_dummy_image_();
    bool create_global_samplers_();
    void destroy_global_samplers_();
    void mark_downstream_dirty_(NodeId root);
    void rebuild_downstream_adj_();

    // Reverse-construction-order tear-down. Tolerates a not-fully-initialised
    // engine and ignores VK_ERROR_DEVICE_LOST. Used by both Engine::shutdown
    // (normal path) and Engine::init's rollback (after a partial init).
    void shutdown_internal_();

    // Bindless slot assignment for a pass (sampled inputs + storage output).
    void assign_bindless_slots_(PassExec& pe);
    void release_bindless_slots_(PassExec& pe);

    // Phase 0: single point of truth for failure reporting. Mirrors to the
    // legacy last_error_/failed_node_id_ fields for backward compat and logs.
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

    Image& output_() { return *output_storage_; }
    Image           dummy_;
    uint32_t        dummy_sampled_slot_ = BindlessTable::INVALID_SLOT;

    ResourceManager resources_;
    // Bindless slot tracking for ResourceManager-owned images.
    // Keyed by ResourceUUID: sampled slot used when read, storage slot when written.
    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> res_sampled_slot_;
    std::unordered_map<ResourceUUID, uint32_t, ResourceUUIDHash> res_storage_slot_;
    // Bindless slots for external uploaded images (image_registry_).
    std::unordered_map<uint64_t, uint32_t> ext_sampled_slot_;

    uint32_t        output_w_ = 512, output_h_ = 512;

    uint64_t compile_generation_   = 0;
    uint64_t pending_generation_   = 0;
    uint64_t installed_generation_ = 0;

    // Param SSBO ring (binding 5 on the bindless set).
    static constexpr uint32_t PARAM_RING = MAX_FRAMES_IN_FLIGHT + 1;
    static_assert(PARAM_RING == BindlessTable::PARAM_RING_SIZE,
        "Engine PARAM_RING must match BindlessTable::PARAM_RING_SIZE");  
    VkBuffer       param_buf_[PARAM_RING]    {};
    VmaAllocation  param_alloc_[PARAM_RING]  {};
    void*          param_mapped_[PARAM_RING] {};
    bool param_dirty_ = false;
    uint32_t       param_write_idx_ = 0;

    // The one global bindless table — set 0, bound forever.
    BindlessTable  bindless_;

    std::unique_ptr<Image> output_storage_;
    Image* output_ptr_ = nullptr;
    struct RetiredImage { std::unique_ptr<Image> img; uint32_t frames_remaining; };
    std::vector<RetiredImage> retired_images_;

    std::vector<PassExec> passes_;
    std::vector<RetiredPass> retired_passes_;
    ResourceUUID  final_output_resource_ = {};

    std::unordered_map<uint64_t, std::unique_ptr<Image>> image_registry_;
    VkImageLayout dummy_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout output_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    std::unordered_map<NodeId, std::vector<NodeId>> downstream_adj_;

    struct PendingPass {
        std::future<CompileResult> fut;
        std::string                name;
        uint32_t                   input_count = 0;
        std::vector<ResourceUUID>  output_resources;
        std::vector<ResourceUUID>  input_resources;
        NodeId                     node_id = 0;
        PassKind                   kind = PassKind::Dispatch;
        ShaderVariantKey           variant_key;
        // Phase 1c: mirror ComputePass::bypassed so the install path can
        // copy it onto the PassExec once the future resolves.
        bool                       bypassed = false;
    };
    std::vector<PendingPass> pending_passes_;
    bool                     pending_active_ = false;
    ResourceUUID             pending_final_output_ = {};

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

    // Lifecycle state machine (see public EngineState enum above).
    std::atomic<EngineState> state_{EngineState::Uninitialized};
    std::mutex               entry_mu_;
    uint64_t          last_presented_generation_{0};
    std::vector<float> last_presented_pixels_;
    uint32_t           last_presented_w_{0};
    uint32_t           last_presented_h_{0};

    DirtySet          dirty_set_;
    uint64_t          exec_generation_ = 0;

    AsyncReadback async_;
};

} // namespace te