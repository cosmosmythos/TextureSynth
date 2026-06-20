# src — C++ Source

## Purpose
All C++20 source for TextureSynth. Three build targets defined in the root `CMakeLists.txt`:
1. `engine` — static lib (no GLFW/ImGui). The core.
2. `texturesynth_core` — nanobind Python extension (the addon's `.pyd`). Built only with `-DBUILD_PYTHON_BINDINGS=ON`.
3. `viewer` — standalone Vulkan window app. Built only with `-BUILD_VIEWER=ON`.

## Ownership
- `src/engine/` — the engine lib (has its own AGENTS.md).
- `src/bindings/bindings.cpp` — the single nanobind source file exposing `Engine`, `Graph`, `NodeLibrary`, `PushConstants`, `EngineError` to Python.
- `src/viewer/` — GLFW + ImGui + RenderDoc app (`main.cpp`, `Window`, `Swapchain`, `Renderer`, `ImGuiLayer`, `RenderDocCapture`).
- `src/viewer/third_party/renderdoc_app.h` — vendored header; NOT linked to renderdoc.dll (loaded at runtime via `GetModuleHandleA`).

## Local Contracts
- **Layering (must hold):**
  `engine` (no deps downward) ← `bindings` (depends on `engine`) ← `viewer` (depends on `engine` + glfw + imgui).
  `bindings` and `viewer` MUST NOT be linked into `engine`. `bindings.cpp` is the only place Python types cross the boundary.
- **Bindings entry pattern** (`bindings.cpp:21-30`): every exposed function does `lock_guard(entry_mutex())` → `check_engine_ready(phase)` → body. Mirrors `TE_GUARD_READY`. Lambdas cannot use the macro, so they call `check_engine_ready` directly.
- **ndarray ownership** (root §5 gotcha): return numpy arrays backed by `nb::capsule(new[], delete[])`. Never back them with a stack `std::vector` — it dangles after return (see `poll_readback` in `bindings.cpp:68`).
- **One VkInstance per process**: both `viewer/main.cpp` and the binding create an `Engine`; only one may be live at a time. `shutdown()` is mandatory before a second `init()`.
- **PushConstants.seed is uint32** (root §5): pass `int` from Python, never `float`.
- **Viewer shader copy**: `CMakeLists.txt:77-81` copies `shader_assets/` next to the exe post-build. Editing shaders in `shader_assets/` is enough; no manual copy.

## Work Guidance
- Before adding a binding: (1) confirm the C++ method exists on `Engine`, (2) add the nanobind def in `bindings.cpp`, (3) mirror the lock+ready pattern.
- The legacy `bindings_pybind11.cpp.bak` is dead — do not revive it; nanobind is the binding layer.
- Viewer is optional and rarely touched. If you must rebuild it, ensure `BUILD_VIEWER=ON` and use `build_fast.bat`.

## Verification
- Engine unit tests (CMake target `engine_tests`) cover the lib — see `src/engine/AGENTS.md` and `tests/AGENTS.md`.
- Binding smoke tests: `tests/python/` exercises `texturesynth_core` from Python.
- `tools/test_new_bindings.py` is a manual binding sanity script (not part of pytest).

## Child DOX Index
- [`engine/AGENTS.md`](engine/AGENTS.md) — Vulkan compute engine core.
  - [`engine/graphfusion/AGENTS.md`](engine/graphfusion/AGENTS.md) — chain fusion subsystem.
