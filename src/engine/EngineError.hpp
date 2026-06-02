#pragma once
#include <cstdint>
#include <string>

namespace te {

using NodeId = uint64_t;

// ---------------------------------------------------------------------------
// EngineError -- structured error channel for the C++/Python product surface.
//
// Every Engine entry point that can fail (init, set_graph, submit_render,
// upload_image, release_image, shutdown) populates a single EngineError
// record reachable via Engine::last_error_record(). The record persists
// until the next successful call clears it.
//
// Code + phase let callers (Blender addon, tests) dispatch on machine-readable
// reasons without parsing log text. The message is for humans; the node id
// pinpoints which node in the graph caused the failure (0 when not applicable).
// ---------------------------------------------------------------------------

enum class EngineErrorCode : uint32_t {
    None = 0,

    // Lifecycle
    InitFailed,           // Vulkan context, AsyncReadback, ImageUploader, samplers, bindless
    ShutdownFailed,       // Should be unreachable; included for completeness

    // Graph submission / compilation
    GraphValidation,      // validate_graph() rejected the Graph
    GraphCompile,         // GraphCompiler::compile() failed (non-shader)
    ShaderCompile,        // Per-pass shader compile failed (failed_node set)
    PipelineCreation,     // VkComputePipeline creation failed

    // Submit / readback
    NoPipeline,           // submit_render called before has_pipeline()
    StaleGeneration,      // submit_render generation no longer current
    SubmitRingFull,       // AsyncReadback ring exhausted (recoverable, retry)

    // Parameter updates
    ParamUnknownNode,     // update_node_params_* called with unknown node id
    ParamUnknownName,     // update_node_params_by_name: unknown param name

    // Image upload
    ImageUploadShape,     // numpy array dims mismatch width/height/4
    ImageUploadRingFull,  // uploader ring exhausted (recoverable, retry)
    ImageUploadOOM,       // vkCreateImage / VMA allocation failed
    ImageReleaseUnknown,  // release_image: no image registered for node

    // Vulkan command-buffer errors (rarely surfaced directly)
    VulkanCommand,        // vkBegin/End/Reset/QueueSubmit failed

    Unknown
};

enum class EnginePhase : uint32_t {
    Idle = 0,
    Init,
    GraphSubmit,          // set_graph -> validate -> kick off async compile
    GraphCompileFinish,   // poll_pending_compiles -> install pipeline
    ParamUpdate,
    ImageUpload,
    ImageRelease,
    Submit,               // submit_render dispatch
    Readback,             // poll_readback
    Shutdown
};

struct EngineError {
    EngineErrorCode code             = EngineErrorCode::None;
    std::string     message;
    NodeId          failed_node      = 0;
    uint64_t        graph_generation = 0;
    EnginePhase     phase            = EnginePhase::Idle;

    bool is_error() const noexcept { return code != EngineErrorCode::None; }
    void clear() noexcept {
        code             = EngineErrorCode::None;
        message.clear();
        failed_node      = 0;
        graph_generation = 0;
        phase            = EnginePhase::Idle;
    }
};

// ---------------------------------------------------------------------------
// Stringification for log lines and Python repr. Used by Engine::set_error_
// to produce readable "[CodeName @ PhaseName]" prefixes.
// ---------------------------------------------------------------------------
inline const char* engine_error_code_name(EngineErrorCode c) noexcept {
    switch (c) {
        case EngineErrorCode::None:                return "None";
        case EngineErrorCode::InitFailed:          return "InitFailed";
        case EngineErrorCode::ShutdownFailed:      return "ShutdownFailed";
        case EngineErrorCode::GraphValidation:     return "GraphValidation";
        case EngineErrorCode::GraphCompile:        return "GraphCompile";
        case EngineErrorCode::ShaderCompile:       return "ShaderCompile";
        case EngineErrorCode::PipelineCreation:    return "PipelineCreation";
        case EngineErrorCode::NoPipeline:          return "NoPipeline";
        case EngineErrorCode::StaleGeneration:     return "StaleGeneration";
        case EngineErrorCode::SubmitRingFull:      return "SubmitRingFull";
        case EngineErrorCode::ParamUnknownNode:    return "ParamUnknownNode";
        case EngineErrorCode::ParamUnknownName:    return "ParamUnknownName";
        case EngineErrorCode::ImageUploadShape:    return "ImageUploadShape";
        case EngineErrorCode::ImageUploadRingFull: return "ImageUploadRingFull";
        case EngineErrorCode::ImageUploadOOM:      return "ImageUploadOOM";
        case EngineErrorCode::ImageReleaseUnknown: return "ImageReleaseUnknown";
        case EngineErrorCode::VulkanCommand:       return "VulkanCommand";
        case EngineErrorCode::Unknown:             return "Unknown";
    }
    return "Unknown";
}

inline const char* engine_phase_name(EnginePhase p) noexcept {
    switch (p) {
        case EnginePhase::Idle:                return "Idle";
        case EnginePhase::Init:                return "Init";
        case EnginePhase::GraphSubmit:         return "GraphSubmit";
        case EnginePhase::GraphCompileFinish:  return "GraphCompileFinish";
        case EnginePhase::ParamUpdate:         return "ParamUpdate";
        case EnginePhase::ImageUpload:         return "ImageUpload";
        case EnginePhase::ImageRelease:        return "ImageRelease";
        case EnginePhase::Submit:              return "Submit";
        case EnginePhase::Readback:            return "Readback";
        case EnginePhase::Shutdown:            return "Shutdown";
    }
    return "Idle";
}

} // namespace te
