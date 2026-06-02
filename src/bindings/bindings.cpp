#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unordered_map.h>

#include <cstring>
#include <memory>

#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PushConstants.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using namespace te;

// ---------------------------------------------------------------------------
// ASYNC API
// ---------------------------------------------------------------------------
// check_engine_ready: throws RuntimeError + populates EngineError when the
// engine is not in the Ready state. Mirrors the C++ TE_GUARD_READY macro
// so Python callers see a uniform error experience across mutators.
//
// IMPORTANT: every bindings function that calls check_engine_ready must
// first take engine.entry_mutex() for the full duration of the call. The
// C++-side TE_GUARD_READY macro does this; the bindings side has to do
// it explicitly because lambdas can't use the macro. The lock pattern
// for every guarded function is:
//
//     std::lock_guard<std::recursive_mutex> lk(engine.entry_mutex());
//     check_engine_ready(engine, EnginePhase::Foo);
//     // ... body ...
//
// Without the lock, a concurrent Engine::shutdown() could tear down
// Vulkan resources between the state check and the body.
// ---------------------------------------------------------------------------
static void check_engine_ready(Engine& engine, EnginePhase phase) {
    if (engine.is_ready()) return;
    const Engine::EngineState s = engine.engine_state();
    engine.set_error_record(EngineError{
        EngineErrorCode::UseAfterShutdown,
        std::string("engine not Ready (state=") + std::to_string((int)s) + ")",
        0, 0, phase});
    throw std::runtime_error("engine not Ready");
}

// submit_render: records + submits a dispatch+copy job on a free ring slot.
// Returns ticket > 0 on success, 0 if all slots busy (retry next tick).
// NEVER blocks the main thread.
static uint64_t submit_render(Engine& engine, PushConstants pc, uint64_t generation) {
    std::lock_guard<std::recursive_mutex> lk(engine.entry_mutex());
    check_engine_ready(engine, EnginePhase::Submit);
    if (!engine.has_pipeline()) {
        engine.set_error_record(EngineError{
            EngineErrorCode::NoPipeline,
            std::string("No pipeline ready. Check has_pipeline() first."),
            0, 0, EnginePhase::Submit});
        throw std::runtime_error("No pipeline ready. Check has_pipeline() first.");
    }
    if (generation != 0) {
        const bool gen_ok = engine.is_generation_ready(generation);
        const bool rev_ok = (generation == engine.current_revision()
                             && engine.has_pipeline());
        if (!gen_ok && !rev_ok) {
            engine.set_error_record(EngineError{
                EngineErrorCode::StaleGeneration,
                std::string("Stale generation/revision (gen=") + std::to_string(generation)
                + ", current=" + std::to_string(engine.installed_generation()) + ")",
                0, generation, EnginePhase::Submit});
            throw std::runtime_error("Stale generation/revision");
        }
    }
    const uint64_t ticket = engine.async_readback().submit(
        engine.ctx(), engine, pc, generation);
    if (ticket == 0) {
        engine.set_error_record(EngineError{
            EngineErrorCode::SubmitRingFull,
            std::string("AsyncReadback ring full -- retry next tick"),
            0, generation, EnginePhase::Submit});
    }
    return ticket;
}

// poll_readback: non-blocking. Returns None if nothing ready,
// else (numpy[H,W,4], generation).
static nb::object poll_readback(Engine& engine) {
    std::lock_guard<std::recursive_mutex> lk(engine.entry_mutex());
    check_engine_ready(engine, EnginePhase::Readback);
    std::vector<float> pixels;
    uint32_t w = 0, h = 0;
    uint64_t gen = 0;
    if (!engine.async_readback().poll(engine.ctx(), pixels, w, h, gen)) {
        return nb::none();
    }
    // Cache last-presented frame for synthetic republish on dirty-skip.
    engine.stash_last_presented(pixels, w, h, gen);

    // Allocate a heap buffer the ndarray will own via nb::capsule.
    // Stack-allocated std::vector data would dangle after this function returns,
    // so we copy into a new[] block whose lifetime is governed by the capsule deleter.
    const size_t count = static_cast<size_t>(h) * w * 4;
    float* data = new float[count];
    std::memcpy(data, pixels.data(), count * sizeof(float));

    nb::capsule owner(data, [](void* p) noexcept {
        delete[] static_cast<float*>(p);
    });

    size_t shape[3] = { static_cast<size_t>(h),
                        static_cast<size_t>(w),
                        static_cast<size_t>(4) };
    nb::ndarray<nb::numpy, float, nb::shape<-1, -1, 4>> arr(
        data, /*ndim=*/3, shape, owner);

    return nb::make_tuple(arr, gen);
}

NB_MODULE(texturesynth_core, m) {
    m.doc() = "TextureSynth Vulkan Engine -- Blender bindings";

    // Phase 0: structured error channel.
    nb::enum_<EngineErrorCode>(m, "EngineErrorCode")
        .value("None",                EngineErrorCode::None)
        .value("InitFailed",          EngineErrorCode::InitFailed)
        .value("ShutdownFailed",      EngineErrorCode::ShutdownFailed)
        .value("UseAfterShutdown",    EngineErrorCode::UseAfterShutdown)
        .value("DoubleInit",          EngineErrorCode::DoubleInit)
        .value("Busy",                EngineErrorCode::Busy)
        .value("InvalidDimensions",   EngineErrorCode::InvalidDimensions)
        .value("GraphValidation",     EngineErrorCode::GraphValidation)
        .value("GraphCompile",        EngineErrorCode::GraphCompile)
        .value("ShaderCompile",       EngineErrorCode::ShaderCompile)
        .value("PipelineCreation",    EngineErrorCode::PipelineCreation)
        .value("NoPipeline",          EngineErrorCode::NoPipeline)
        .value("StaleGeneration",     EngineErrorCode::StaleGeneration)
        .value("SubmitRingFull",      EngineErrorCode::SubmitRingFull)
        .value("ParamUnknownNode",    EngineErrorCode::ParamUnknownNode)
        .value("ParamUnknownName",    EngineErrorCode::ParamUnknownName)
        .value("ImageUploadShape",    EngineErrorCode::ImageUploadShape)
        .value("ImageUploadRingFull", EngineErrorCode::ImageUploadRingFull)
        .value("ImageUploadOOM",      EngineErrorCode::ImageUploadOOM)
        .value("ImageReleaseUnknown", EngineErrorCode::ImageReleaseUnknown)
        .value("VulkanCommand",       EngineErrorCode::VulkanCommand)
        .value("Unknown",             EngineErrorCode::Unknown);

    nb::enum_<EnginePhase>(m, "EnginePhase")
        .value("Idle",               EnginePhase::Idle)
        .value("Init",               EnginePhase::Init)
        .value("GraphSubmit",        EnginePhase::GraphSubmit)
        .value("GraphCompileFinish", EnginePhase::GraphCompileFinish)
        .value("ParamUpdate",        EnginePhase::ParamUpdate)
        .value("ImageUpload",        EnginePhase::ImageUpload)
        .value("ImageRelease",       EnginePhase::ImageRelease)
        .value("Submit",             EnginePhase::Submit)
        .value("Readback",           EnginePhase::Readback)
        .value("Shutdown",           EnginePhase::Shutdown);

    nb::class_<EngineError>(m, "EngineError")
        .def(nb::init<>())
        .def_rw("code",             &EngineError::code)
        .def_rw("message",          &EngineError::message)
        .def_rw("failed_node",      &EngineError::failed_node)
        .def_rw("graph_generation", &EngineError::graph_generation)
        .def_rw("phase",            &EngineError::phase)
        .def("is_error", &EngineError::is_error);

    m.def("set_log_callback", [](nb::object callback) {
        if (callback.is_none()) {
            te::set_log_sink(nullptr);
            return;
        }
        auto cb = std::make_shared<nb::object>(std::move(callback));
        te::set_log_sink([cb](const char* level, const std::string& message) {
            nb::gil_scoped_acquire gil;
            try {
                // Decode as UTF-8 with replacement to survive shaderc's
                // non-UTF-8 error messages (Windows code page bytes).
                nb::object py_msg = nb::steal<nb::object>(
                    PyUnicode_DecodeUTF8(message.data(),
                                         (Py_ssize_t)message.size(),
                                         "replace"));
                if (!py_msg.is_valid()) { PyErr_Clear(); return; }
                (*cb)(level, py_msg);
            } catch (nb::python_error& e) {
                e.restore();
                PyErr_WriteUnraisable(Py_None);
            }
        });
    }, "callback"_a);

    nb::class_<PushConstants>(m, "PushConstants")
        .def(nb::init<>())
        .def_rw("resolution_x", &PushConstants::resolution_x)
        .def_rw("resolution_y", &PushConstants::resolution_y)
        .def_rw("seed",         &PushConstants::seed)
        .def_rw("time",         &PushConstants::time);

    nb::enum_<ChannelFormat>(m, "ChannelFormat")
        .value("Mono",     ChannelFormat::Mono)
        .value("UV",       ChannelFormat::UV)
        .value("RGB",      ChannelFormat::RGB)
        .value("RGBA",     ChannelFormat::RGBA)
        .value("ID",       ChannelFormat::ID)
        .value("Metadata", ChannelFormat::Metadata);

    nb::class_<Graph>(m, "Graph")
        .def(nb::init<>())
        .def("add_node",
             [](Graph& g, uint64_t id, const std::string& type,
                ChannelFormat format_override,
                const std::string& debug_name,
                bool muted, bool bypassed) {
                 NodeInstance ni;
                 ni.id              = id;
                 ni.type_id         = type;
                 ni.format_override = format_override;
                 ni.debug_name      = debug_name;
                 ni.muted           = muted;
                 ni.bypassed        = bypassed;
                 g.nodes.push_back(std::move(ni));
             },
             "id"_a, "type"_a,
             "format_override"_a = ChannelFormat::RGBA,
             "debug_name"_a = std::string(),
             "muted"_a = false,
             "bypassed"_a = false)
        .def("add_connection",
             [](Graph& g, uint64_t src_node, uint32_t src_sock,
                          uint64_t dst_node, uint32_t dst_sock) {
                 g.connections.push_back({src_node, src_sock, dst_node, dst_sock});
             })
        .def("set_output",
             [](Graph& g, uint64_t id) {
                 g.output_node = id;
             });

    nb::enum_<SocketType>(m, "SocketType")
        .value("Float",     SocketType::Float)
        .value("Vec4",      SocketType::Vec4)
        .value("Sampler2D", SocketType::Sampler2D);

    nb::class_<Socket>(m, "Socket")
        .def(nb::init<>())
        .def_rw("name",   &Socket::name)
        .def_rw("type",   &Socket::type)
        .def_rw("format", &Socket::format);

    nb::class_<NodeParam>(m, "NodeParam")
        .def(nb::init<>())
        .def_rw("name",          &NodeParam::name)
        .def_rw("display_name",  &NodeParam::display_name)
        .def_rw("description",   &NodeParam::description)
        .def_rw("default_value", &NodeParam::default_value)
        .def_rw("min_value",     &NodeParam::min_value)
        .def_rw("max_value",     &NodeParam::max_value)
        .def_rw("step",          &NodeParam::step)
        .def_rw("is_integer",    &NodeParam::is_integer)
        .def_rw("as_socket",     &NodeParam::as_socket);

    nb::class_<NodeType>(m, "NodeType")
        .def(nb::init<>())
        .def_rw("id",            &NodeType::id)
        .def_rw("display_name",  &NodeType::display_name)
        .def_rw("description",   &NodeType::description)
        .def_rw("inputs",        &NodeType::inputs)
        .def_rw("outputs",       &NodeType::outputs)
        .def_rw("params",        &NodeType::params);

    nb::class_<NodeLibrary>(m, "NodeLibrary")
        .def("all", &NodeLibrary::all, nb::rv_policy::reference);

    nb::class_<Engine>(m, "Engine")
        .def(nb::init<>())

        .def("node_library", &Engine::node_library, nb::rv_policy::reference)

        // Call once from register() on Blender's main thread.
        .def("init",
            [](Engine& e,
               bool enable_validation,
               const std::string& cache_dir,
               const std::string& nodes_dir,
               const std::string& glsl_dir) -> bool {
                return e.init(VK_NULL_HANDLE, nullptr, 0,
                              enable_validation, cache_dir,
                              nodes_dir, glsl_dir);
            },
            "enable_validation"_a = false,
            "cache_dir"_a = "shader_cache",
            "nodes_dir"_a = "",
            "glsl_dir"_a  = "")

        // Call from unregister() on Blender's main thread.
        .def("shutdown", &Engine::shutdown)

        // Submit a new graph for async compilation. Returns generation id, or 0 on failure.
        .def("set_graph", [](Engine& e, const Graph& g) -> uint64_t {
            std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
            check_engine_ready(e, EnginePhase::GraphSubmit);
            return e.set_graph(g);
        })

        // Call once per timer tick (main thread). Installs pipeline if compile finished.
        .def("poll_pending_compiles", [](Engine& e) {
            std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
            check_engine_ready(e, EnginePhase::GraphCompileFinish);
            e.poll_pending_compiles();
        })

        // True when a pipeline is ready to dispatch.
        .def("has_pipeline", &Engine::has_pipeline)
        .def("compile_generation", &Engine::compile_generation)
        .def("installed_generation", &Engine::installed_generation)
        .def("current_revision", &Engine::current_revision)
        .def("resource_count", [](Engine& e) {
            // Count of live per-node images held by ResourceManager.
            // Cheap O(1) via internal map size -- exposed for Python diagnostics.
            return (int)e.resources_live_count();
        })
        .def("resource_bytes", [](Engine& e) {
            return (uint64_t)e.resources().current_bytes();
        })
        .def("resource_budget_bytes", [](Engine& e) {
            return (uint64_t)e.resources().budget_bytes();
        })
        .def("set_memory_budget_mb", [](Engine& e, size_t mb) {
            std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
            check_engine_ready(e, EnginePhase::Idle);
            e.resources().set_memory_budget_mb(mb);
        })
        .def("is_generation_ready", &Engine::is_generation_ready)
        .def("is_ready", &Engine::is_ready)
        .def("engine_state", [](const Engine& e) {
            return (int)e.engine_state();
        })
        .def("set_precision", &Engine::set_precision)
        .def("precision", &Engine::precision)
        .def("set_resolution", [](Engine& e, uint32_t w, uint32_t h) {
            std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
            check_engine_ready(e, EnginePhase::Idle);
            e.set_resolution(w, h);
        })

        // Last compile/graph error string.
        .def("last_error", [](const Engine& e) {
            const std::string& s = e.last_error();
            return nb::steal<nb::object>(
                PyUnicode_DecodeUTF8(s.data(), (Py_ssize_t)s.size(), "replace"));
        })
        .def("failed_node", &Engine::failed_node)

        // Phase 0: structured error channel.
        .def("last_error_record", &Engine::last_error_record, nb::rv_policy::reference)
        .def("clear_error",       &Engine::clear_error)

        // Upload node slider values into the GPU parameter SSBO. Call before dispatch.
        // Replaces the old push-constant hot_params approach -- no 28-float limit.
        .def("update_node_params_by_id",
            [](Engine& e, uint64_t id, const std::vector<float>& p) {
                std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
                check_engine_ready(e, EnginePhase::ParamUpdate);
                e.update_node_params_by_id(id, p);
            })

        .def("update_node_params_by_name",
            [](Engine& e, uint64_t id,
               const std::unordered_map<std::string, float>& kv) {
                std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
                check_engine_ready(e, EnginePhase::ParamUpdate);
                e.update_node_params_by_name(id, kv);
            },
            "node_id"_a, "params"_a)

        .def("upload_image",
            [](Engine& e, uint64_t node_id,
               nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> pixels,
               uint32_t width, uint32_t height) -> bool {
                std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
                check_engine_ready(e, EnginePhase::ImageUpload);
                if (pixels.shape(0) != height ||
                    pixels.shape(1) != width  ||
                    pixels.shape(2) != 4) {
                    return false;
                }
                return e.upload_image(node_id, pixels.data(), width, height);
            })
        .def("release_image", [](Engine& e, uint64_t node_id) -> bool {
            std::lock_guard<std::recursive_mutex> lk(e.entry_mutex());
            check_engine_ready(e, EnginePhase::ImageRelease);
            return e.release_image(node_id);
        })

        .def("param_layout", [](const Engine& e) {
            return e.param_layout();  // dict {id:int -> base:int}
        })

        .def("total_param_floats", &Engine::total_param_floats)

        // Dispatch + CPU readback. Only call after has_pipeline() == True.
        // MUST be called from Blender's main thread (timer callback). Never from a Python thread.
        .def("submit_render", &submit_render,
             "pc"_a, "generation"_a = 0,
             "Async: submits a render job. Returns ticket (>0) or 0 if ring full.")
        .def("poll_readback", &poll_readback,
             "Async: returns (pixels, generation) tuple if a job finished, else None.")
        .def("async_in_flight", [](Engine& e) {
             return e.async_readback().any_in_flight();
        });
}
