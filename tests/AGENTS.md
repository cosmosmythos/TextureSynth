# tests — C++ gtest + Python pytest

## Purpose
Two independent test suites:
1. **C++ gtest** (`engine_tests` target) — exercises the `engine` lib directly (no Python). Covers Vulkan context, graph validation, full pipeline, async readback, dirty set, aliasing, mask/mute nodes, timestamps, combine/separate RGBA, image upload, noise nodes, simplex debug, the full fusion path, and the cross-group chain preview repro (`test_repro_blend_preview.cpp`).
2. **Python pytest** (`tests/python/`) — exercises the `texturesynth_core` nanobind binding the way the Blender addon uses it.

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
- **Validation**: C++ tests compile with `TS_TESTS_ENABLE_VALIDATION=1` (see `tests/CMakeLists.txt:43`) — Vulkan validation layers are active in the test build.
- **One VkInstance per process** (root §5): the Python session fixture creates exactly one `Engine` and shuts it down at teardown. C++ tests should construct/destroy their own `Engine` per case and call `shutdown()`.

## Work Guidance
- Prefer adding to an existing test file when the topic fits (e.g. noise behavior → `test_noise_nodes.cpp`) rather than spawning a new file.
- Keep heavy/large debug binaries out of the suite — `test_simplex_debug.cpp` (~63k) and `test_full_pipeline.cpp` are already at the upper end.
- Cache dirs: Python uses `tests/python/cache/shader` (gitignored). Do not point tests at `build/_deps/` (root §2 cache trap).

## Verification
- C++: `ctest --test-dir build -C Release` (or run `engine_tests` directly). GoogleTest discovery is wired via `gtest_discover_tests` in `tests/CMakeLists.txt:48`.
- Python: `pytest tests/python/` from repo root. Config in repo-root `pytest.ini`.

## Child DOX Index
None — `tests/python/` is small and cohesive; this doc covers both suites. If a third suite (e.g. shader-only glslViewer automation) is added, split a child doc.
