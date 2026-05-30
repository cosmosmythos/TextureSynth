#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>

#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/Logging.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PushConstants.hpp"

namespace py = pybind11;
using namespace te;

// ---------------------------------------------------------------------------
// ASYNC API
// ---------------------------------------------------------------------------
// submit_render: records + submits a dispatch+copy job on a free ring slot.
// Returns ticket > 0 on success, 0 if all slots busy (retry next tick).
// NEVER blocks the main thread.
static uint64_t submit_render(Engine& engine, PushConstants pc, uint64_t generation) {
    if (!engine.has_pipeline()) {
        throw std::runtime_error("No pipeline ready. Check has_pipeline() first.");
    }
    if (generation != 0) {
        const bool gen_ok = engine.is_generation_ready(generation);
        const bool rev_ok = (generation == engine.current_revision()
                             && engine.has_pipeline());
        if (!gen_ok && !rev_ok) {
            throw std::runtime_error("Stale generation/revision");
        }
    }
    return engine.async_readback().submit(engine.ctx(), engine, pc, generation);
}

// poll_readback: non-blocking. Returns None if nothing ready,
// else (numpy[H,W,4], generation).
static py::object poll_readback(Engine& engine) {
    std::vector<float> pixels;
    uint32_t w = 0, h = 0;
    uint64_t gen = 0;
    if (!engine.async_readback().poll(engine.ctx(), pixels, w, h, gen)) {
        return py::none();
    }
    // Cache last-presented frame for synthetic republish on dirty-skip.
    engine.stash_last_presented(pixels, w, h, gen);
    py::array_t<float> arr({
        static_cast<py::ssize_t>(h),
        static_cast<py::ssize_t>(w),
        static_cast<py::ssize_t>(4)
    });
    std::memcpy(arr.mutable_data(), pixels.data(), pixels.size() * sizeof(float));
    return py::make_tuple(std::move(arr), gen);
}

PYBIND11_MODULE(texturesynth_core, m) {
    m.doc() = "TextureSynth Vulkan Engine — Blender bindings";

    m.def("set_log_callback", [](py::object callback) {
        if (callback.is_none()) {
            te::set_log_sink(nullptr);
            return;
        }
        auto cb = std::make_shared<py::object>(std::move(callback));
        te::set_log_sink([cb](const char* level, const std::string& message) {
            py::gil_scoped_acquire gil;
            try {
                // Decode as UTF-8 with replacement to survive shaderc's
                // non-UTF-8 error messages (Windows code page bytes).
                py::object py_msg = py::reinterpret_steal<py::object>(
                    PyUnicode_DecodeUTF8(message.data(),
                                         (Py_ssize_t)message.size(),
                                         "replace"));
                if (!py_msg) { PyErr_Clear(); return; }
                (*cb)(level, py_msg);
            } catch (py::error_already_set& e) {
                e.restore();
                PyErr_WriteUnraisable(Py_None);
            }
        });
    }, py::arg("callback"));

    py::class_<PushConstants>(m, "PushConstants")
        .def(py::init<>())
        .def_readwrite("resolution_x", &PushConstants::resolution_x)
        .def_readwrite("resolution_y", &PushConstants::resolution_y)
        .def_readwrite("seed",         &PushConstants::seed)
        .def_readwrite("time",         &PushConstants::time);

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_node",       [](Graph& g, uint64_t id, const std::string& type, ChannelFormat format_override) {
            g.nodes.push_back({id, type, format_override});
        }, py::arg("id"), py::arg("type"), py::arg("format_override") = ChannelFormat::RGBA)
        .def("add_connection", [](Graph& g, uint64_t src_node, uint32_t src_sock,
                                           uint64_t dst_node, uint32_t dst_sock) {
            g.connections.push_back({src_node, src_sock, dst_node, dst_sock});
        })
        .def("set_output", [](Graph& g, uint64_t id) {
            g.output_node = id;
        });

    py::enum_<SocketType>(m, "SocketType")
        .value("Float", SocketType::Float)
        .value("Vec4", SocketType::Vec4)
        .value("Sampler2D", SocketType::Sampler2D)
        .export_values();

    py::enum_<ChannelFormat>(m, "ChannelFormat")
        .value("Mono", ChannelFormat::Mono)
        .value("UV", ChannelFormat::UV)
        .value("RGB", ChannelFormat::RGB)
        .value("RGBA", ChannelFormat::RGBA)
        .value("ID", ChannelFormat::ID)
        .value("Metadata", ChannelFormat::Metadata)
        .export_values();

    py::class_<Socket>(m, "Socket")
        .def(py::init<>())
        .def_readwrite("name",   &Socket::name)
        .def_readwrite("type",   &Socket::type)
        .def_readwrite("format", &Socket::format);

    py::class_<NodeParam>(m, "NodeParam")
        .def(py::init<>())
        .def_readwrite("name",          &NodeParam::name)
        .def_readwrite("display_name",  &NodeParam::display_name)
        .def_readwrite("description",   &NodeParam::description)
        .def_readwrite("default_value", &NodeParam::default_value)
        .def_readwrite("min_value",     &NodeParam::min_value)
        .def_readwrite("max_value",     &NodeParam::max_value)
        .def_readwrite("step",          &NodeParam::step)
        .def_readwrite("is_integer",    &NodeParam::is_integer)
        .def_readwrite("as_socket",     &NodeParam::as_socket);
    
    py::class_<NodeType>(m, "NodeType")
        .def(py::init<>())
        .def_readwrite("id",            &NodeType::id)
        .def_readwrite("display_name",  &NodeType::display_name)
        .def_readwrite("description",   &NodeType::description)
        .def_readwrite("inputs",        &NodeType::inputs)
        .def_readwrite("outputs",       &NodeType::outputs)
        .def_readwrite("params",        &NodeType::params);

    py::class_<NodeLibrary>(m, "NodeLibrary")
        .def("all", &NodeLibrary::all, py::return_value_policy::reference);

    py::class_<Engine>(m, "Engine")
        .def(py::init<>())
        
        .def("node_library", &Engine::node_library, py::return_value_policy::reference)

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
            py::arg("enable_validation") = false,
            py::arg("cache_dir") = "shader_cache",
            py::arg("nodes_dir") = "",
            py::arg("glsl_dir")  = "")

        // Call from unregister() on Blender's main thread.
        .def("shutdown", &Engine::shutdown)

        // Submit a new graph for async compilation. Returns generation id, or 0 on failure.
        .def("set_graph", [](Engine& e, const Graph& g) -> uint64_t {
            return e.set_graph(g);
        })

        // Call once per timer tick (main thread). Installs pipeline if compile finished.
        .def("poll_pending_compiles", &Engine::poll_pending_compiles)

        // True when a pipeline is ready to dispatch.
        .def("has_pipeline", &Engine::has_pipeline)
        .def("compile_generation", &Engine::compile_generation)
        .def("installed_generation", &Engine::installed_generation)
        .def("current_revision", &Engine::current_revision)
        .def("resource_count", [](Engine& e) {
            // Count of live per-node images held by ResourceManager.
            // Cheap O(1) via internal map size — exposed for Python diagnostics.
            return (int)e.resources_live_count();
        })
        .def("resource_bytes", [](Engine& e) {
            return (uint64_t)e.resources().current_bytes();
        })
        .def("resource_budget_bytes", [](Engine& e) {
            return (uint64_t)e.resources().budget_bytes();
        })
        .def("set_memory_budget_mb", [](Engine& e, size_t mb) {
            e.resources().set_memory_budget_mb(mb);
        })    
        .def("is_generation_ready", &Engine::is_generation_ready)
        .def("set_precision", &Engine::set_precision)
        .def("precision", &Engine::precision)
        .def("set_resolution", &Engine::set_resolution)

        // Last compile/graph error string.
        .def("last_error", [](const Engine& e) {
            const std::string& s = e.last_error();
            return py::reinterpret_steal<py::object>(
                PyUnicode_DecodeUTF8(s.data(), (Py_ssize_t)s.size(), "replace"));
        })
        .def("failed_node", &Engine::failed_node)

        // Upload node slider values into the GPU parameter SSBO. Call before dispatch.
        // Replaces the old push-constant hot_params approach — no 28-float limit.
        .def("update_node_params_by_id",
            [](Engine& e, uint64_t id, const std::vector<float>& p) {
                e.update_node_params_by_id(id, p);
            })

        .def("update_node_params_by_name",
            [](Engine& e, uint64_t id,
               const std::unordered_map<std::string, float>& kv) {
                e.update_node_params_by_name(id, kv);
            },
            py::arg("node_id"), py::arg("params"))

        .def("upload_image", [](Engine& e, uint64_t node_id, py::array_t<float> pixels, uint32_t width, uint32_t height) -> bool {
            py::buffer_info buf = pixels.request();
            if (buf.size < static_cast<py::ssize_t>(width * height * 4)) {
                return false;
            }
            return e.upload_image(node_id, static_cast<const float*>(buf.ptr), width, height);
        })
        .def("release_image", &Engine::release_image)

       .def("param_layout", [](const Engine& e) {
           return e.param_layout();  // dict {id:int -> base:int}
       })

       .def("total_param_floats", &Engine::total_param_floats)

        // Dispatch + CPU readback. Only call after has_pipeline() == True.
        // MUST be called from Blender's main thread (timer callback). Never from a Python thread.
        .def("submit_render", &submit_render,
             py::arg("pc"), py::arg("generation") = 0,
             "Async: submits a render job. Returns ticket (>0) or 0 if ring full.")
        .def("poll_readback", &poll_readback,
             "Async: returns (pixels, generation) tuple if a job finished, else None.")
        .def("async_in_flight", [](Engine& e) {
             return e.async_readback().any_in_flight();
        });
}
