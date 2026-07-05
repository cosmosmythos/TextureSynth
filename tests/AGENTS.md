# tests — C++ gtest + Python pytest

## Purpose
Two independent test suites:
1. **C++ gtest** (`engine_tests` target) — exercises the `engine` lib directly (no Python). Covers Vulkan context, graph validation (incl. inclusive-IR contract), full pipeline, async readback, dirty set, aliasing, mask/mute nodes, timestamps, combine/separate RGBA, image upload, noise node smoke tests, the full fusion path, and the cross-group chain preview repro (`test_repro_blend_preview.cpp`).
2. **Python pytest** (`tests/python/`) — exercises the `texturesynth_core` nanobind binding the way the Blender addon uses it. Includes durable regression guards: `test_claim_verification.py` (inclusive IR, param SSBO seeding, mask modulation), `test_fused_wiring_key.py` (swapped A/B internal wiring must produce different output), and `test_addon_preview_root.py` (addon preview-node selection, mocked bpy).

## Ownership
- Root `tests/*.cpp` + `tests/*.hpp` — gtest sources and shared asset helpers.
- `tests/CMakeLists.txt` — declares `engine_tests`, fetches GoogleTest v1.14.0 via FetchContent.
- `tests/python/` — pytest suite + `conftest.py` (session fixtures: `assets`, `engine`).
- `tests/texturesynth_test.png`, `tests/inspect_perlin_invert.cpp` — fixtures/inspection helpers.
- `tests/compare_fusion_outputs.bat` — manual fused-vs-unfused diff helper.

## Local Contracts
- **C++ tests are OFF by default** (root §5 gotcha). Enable with `cmake -S . -B build -DBUILD_TESTS:BOOL=ON`, then `cmake --build build --config Release --target engine_tests`. Never `--target clean`.
- **New C++ test file**: append the `.cpp` to the `add_executable(engine_tests ...)` list in `tests/CMakeLists.txt` (there is no globbing).
- **Python tests auto-skip on no-GPU**: `conftest.py` calls `Engine.init()`; if Vulkan init fails the whole session skips (not fails). Do not remove this guard — headless CI without a GPU depends on it.
- **Python import path**: `conftest.py` inserts `build/Release` into `sys.path` so `import texturesynth_core` works without installing the wheel. The binding must be built first (`-DBUILD_PYTHON_BINDINGS=ON`).
- **Asset paths**: `TEXTURESYNTH_TEST_ASSET_DIR` (C++) and `assets` fixture (Python) resolve to the repo root. Tests read `shader_assets/{nodes,glsl}` and `tests/texturesynth_test.png`.
- **Validation**: C++ tests compile with `TS_TESTS_ENABLE_VALIDATION=1` (see `tests/CMakeLists.txt:58`) — Vulkan validation layers are active in the test build.
- **One VkInstance per process** (root §5): the Python session fixture creates exactly one `Engine` and shuts it down at teardown. C++ tests should construct/destroy their own `Engine` per case and call `shutdown()`.

## Work Guidance
- Prefer adding to an existing test file when the topic fits (e.g. noise behavior → `test_noise_nodes.cpp`) rather than spawning a new file.
- Keep heavy/large debug binaries out of the suite — `test_full_pipeline.cpp` is at the upper end. Historical one-off debug dumps (e.g. the old `test_simplex_debug.cpp` tiling investigation, `test_tiling_math.cpp`, `test_blend_ssbodump.cpp`) have been removed; do not reintroduce them. Tiling correctness is verified in `shader_assets/glsl/noise_common.glsl` (GLSL-spec `mod()`), not re-tested here.
- Cache dirs: Python uses `tests/python/cache/shader` (gitignored). Do not point tests at `build/_deps/` (root §2 cache trap).
- **Engine API ordering**: `set_resolution(w, h)` MUST be called BEFORE `set_graph()`. Group output images are allocated at `set_graph()` time using the current `output_w_`/`output_h_`. Tests that call `set_graph()` first will get 512x512 output images and dispatch at the wrong resolution — returning black.

## Verification
- C++: `ctest --test-dir build -C Release` (or run `engine_tests` directly). GoogleTest discovery is wired via `gtest_discover_tests` in `tests/CMakeLists.txt:63`. All 85 tests pass.
- Python: `pytest tests/python/` from repo root. Config in repo-root `pytest.ini`.

## Child DOX Index
None — `tests/python/` is small and cohesive; this doc covers both suites. If a third suite (e.g. shader-only glslViewer automation) is added, split a child doc.
