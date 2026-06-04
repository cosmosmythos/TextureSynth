#pragma once
// Test-only helper: resolve the shader_assets directory from any working
// directory. The default behavior (passed through to Engine::init) is
// "shader_assets/nodes" relative to the CWD, which works when tests are
// run from the project root but breaks when a launcher (RenderDoc,
// CTest with a different cwd, etc.) invokes the binary.
//
// Resolution order:
//   1. TEXTURESYNTH_TEST_ASSET_DIR env var (explicit override).
//   2. CMake-defined TEXTURESYNTH_TEST_ASSET_DIR macro (set in
//      tests/CMakeLists.txt to ${CMAKE_SOURCE_DIR} at configure time).
//   3. Walk up from the executable's directory, looking for a
//      `shader_assets/nodes` subdirectory. This is the most robust
//      path -- it works from any CWD because the binary knows its own
//      location.
//   4. CWD-relative fallback ("shader_assets/nodes"). This is the
//      original behavior, kept as a last-resort so manual `cd && run`
//      workflows don't break.
//
// All paths returned are absolute (via std::filesystem::absolute) so
// downstream code never has to reason about CWD.
#include "test_assets_impl.hpp"
