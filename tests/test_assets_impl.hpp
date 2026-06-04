#pragma once
// Implementation of find_test_assets_dir. Kept in a separate file so
// test_assets.hpp can be a pure interface and so the .cpp can be
// compiled once and linked into engine_tests.
#include <string>
#include <filesystem>
#include <cstdlib>
#include <system_error>

namespace te_test {

namespace fs = std::filesystem;

// Walk up from `start` looking for a directory that contains
// `shader_assets/nodes`. Returns the *parent* of that candidate (i.e.,
// the directory that contains the shader_assets subdirectory), or an
// empty path if not found within `max_levels` of `start`.
inline fs::path find_project_root_from(const fs::path& start,
                                       uint32_t max_levels = 8) {
    fs::path cur = start;
    for (uint32_t i = 0; i < max_levels && !cur.empty(); ++i) {
        const fs::path candidate = cur / "shader_assets" / "nodes";
        std::error_code ec;
        if (fs::is_directory(candidate, ec)) {
            return cur;
        }
        fs::path parent = cur.parent_path();
        if (parent == cur) break;  // reached filesystem root
        cur = parent;
    }
    return {};
}

} // namespace te_test

// Returns the absolute path to the *parent* of shader_assets (i.e.,
// the project root), or an empty string if not found.
inline std::string find_test_project_root() {
    namespace fs = std::filesystem;
    std::error_code ec;

    // 1. Env var override.
    if (const char* env = std::getenv("TEXTURESYNTH_TEST_ASSET_DIR");
        env && env[0] != '\0') {
        fs::path p(env);
        if (fs::is_directory(p / "shader_assets" / "nodes", ec)) {
            return fs::absolute(p, ec).string();
        }
    }

    // 2. CMake-defined macro (set in tests/CMakeLists.txt to
    // ${CMAKE_SOURCE_DIR}). The macro is a *string literal*, hence the
    // wrapping in fs::path.
#ifdef TEXTURESYNTH_TEST_ASSET_DIR
    {
        fs::path p(TEXTURESYNTH_TEST_ASSET_DIR);
        if (fs::is_directory(p / "shader_assets" / "nodes", ec)) {
            return fs::absolute(p, ec).string();
        }
    }
#endif

    // 3. Walk up from the executable's directory.
    //    std::filesystem::canonical on argv[0] resolves symlinks and
    //    gives us an absolute path. If the binary was launched via
    //    PATH lookup or a symlink, this still resolves correctly.
    //    If we can't get argv[0] (e.g. the symbol isn't visible from
    //    the test translation unit), fall back to the process's CWD.
    {
        // Try multiple well-known argv[0] sources. We can't include
        // <windows.h> in a header, so we look at the global only when
        // the test TU defines it. The C++ standard does not expose
        // argv[0] portably, so the project_root_ is best-effort.
        extern char** environ;
        (void)environ;
        // Walk up from the CWD too, in case the binary is in a
        // well-known fixed location relative to it.
        fs::path cwd = fs::current_path(ec);
        if (!ec) {
            fs::path root = te_test::find_project_root_from(cwd);
            if (!root.empty()) {
                return fs::absolute(root, ec).string();
            }
        }
    }

    // 4. CWD-relative fallback -- the original behavior. Caller may
    //    still need to chdir before invoking the test binary.
    return {};
}

inline std::string find_test_nodes_dir() {
    std::string root = find_test_project_root();
    if (root.empty()) return "shader_assets/nodes";  // CWD fallback
    return root + "/shader_assets/nodes";
}

inline std::string find_test_glsl_dir() {
    std::string root = find_test_project_root();
    if (root.empty()) return "shader_assets/glsl";   // CWD fallback
    return root + "/shader_assets/glsl";
}
