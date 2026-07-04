cmake_minimum_required(VERSION 3.24)
project(texture_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:preprocessor /MP)
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Vulkan SDK (required for shaderc + headers)
find_package(Vulkan REQUIRED COMPONENTS shaderc_combined)

include(cmake/Dependencies.cmake)
include(FetchContent)
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0)
FetchContent_MakeAvailable(json)

# ---- Engine static library (no GLFW, no ImGui) ----
add_library(engine STATIC
    src/engine/Engine.cpp
    src/engine/VulkanContext.cpp
    src/engine/ShaderCompiler.cpp
    src/engine/ShaderCache.cpp
    src/engine/ComputePipeline.cpp
    src/engine/Image.cpp
    src/engine/NodeLibrary.cpp
    src/engine/NodeRegistryLoader.cpp
    src/engine/GraphCompiler.cpp
    src/engine/GraphIR.cpp
    src/engine/ResourceManager.cpp
    src/engine/AsyncReadback.cpp
    src/engine/ImageUploader.cpp
    src/engine/BindlessTable.cpp
    src/engine/EngineBarriers.cpp
    src/engine/EngineDispatch.cpp
    src/engine/EngineParams.cpp
    src/engine/EnginePassCompile.cpp
    src/engine/EngineGroupCompile.cpp
    src/engine/StorageFormat.cpp

    src/engine/register_allocation/GraphColorer.cpp
    src/engine/register_allocation/InterferenceGraph.cpp
    src/engine/memory_allocation/AliasColorer.cpp
)
target_include_directories(engine PUBLIC src)
target_link_libraries(engine PUBLIC
    Vulkan::Vulkan
    Vulkan::shaderc_combined
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    nlohmann_json::nlohmann_json
)

# ---- Viewer executable (optional) ----
option(BUILD_VIEWER "Build standalone viewer" OFF)
if(BUILD_VIEWER)
add_executable(viewer
    src/viewer/main.cpp
    src/viewer/Window.cpp
    src/viewer/Swapchain.cpp
    src/viewer/ImGuiLayer.cpp
    src/viewer/Renderer.cpp
    src/viewer/RenderDocCapture.cpp
)
# RenderDoc capture API is a header-only library at src/viewer/third_party.
# We do NOT link to renderdoc.dll at build time -- the DLL is found at
# runtime via GetModuleHandleA when qrenderdoc attaches to the process.
# This keeps the viewer binary usable with zero RenderDoc dependency.
target_include_directories(viewer PRIVATE src/viewer/third_party)
target_link_libraries(viewer PRIVATE engine glfw imgui)

# Copy shaders next to executable
add_custom_command(TARGET viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shader_assets
        $<TARGET_FILE_DIR:viewer>/shader_assets
)
endif()

# ---- Python Extension (TextureSynth Addon) ----
option(BUILD_PYTHON_BINDINGS "Build nanobind extension" OFF)
if(BUILD_PYTHON_BINDINGS)
    # nanobind requires find_package(Python ...) with Development (not Development.Module
    # alone) per its config error message. Accept the Python3_* hints passed by
    # build_whl_both.bat so the right interpreter is locked in.
    if(NOT Python_FOUND)
        if(DEFINED Python3_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
            set(Python_EXECUTABLE   "${Python3_EXECUTABLE}")
        endif()
        if(DEFINED Python3_LIBRARY AND NOT DEFINED Python_LIBRARY)
            set(Python_LIBRARY      "${Python3_LIBRARY}")
        endif()
        if(DEFINED Python3_INCLUDE_DIR AND NOT DEFINED Python_INCLUDE_DIR)
            set(Python_INCLUDE_DIR  "${Python3_INCLUDE_DIR}")
        endif()
        find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development)
    endif()

    # Locate nanobind. Two viable strategies:
    #   1) pip-installed nanobind:  python -m nanobind --cmake_dir
    #   2) FetchContent_Declare:    pulls from GitHub
    # We try (1) first (fast, version-pinned by the active interpreter), fall back to (2).
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_VARIABLE _nb_cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nb_pip_rc)
    if(_nb_pip_rc EQUAL 0 AND _nb_cmake_dir)
        message(STATUS "Found nanobind via pip at: ${_nb_cmake_dir}")
        set(nanobind_DIR "${_nb_cmake_dir}")
        find_package(nanobind CONFIG REQUIRED)
    else()
        message(STATUS "nanobind not found via pip; falling back to FetchContent")
        include(FetchContent)
        FetchContent_Declare(nanobind
            GIT_REPOSITORY https://github.com/wjakob/nanobind.git
            GIT_TAG v2.12.0)
        FetchContent_MakeAvailable(nanobind)
    endif()

    # NOMINSIZE: don't override our /O2 with -Os; engine dwarfs binding overhead.
    # STABLE_ABI deferred until Python 3.12+ is the floor (Blender 5.0 ships 3.13 ABI-stable).
    nanobind_add_module(texturesynth_core NOMINSIZE src/bindings/bindings.cpp)
    target_link_libraries(texturesynth_core PRIVATE engine)
endif()

# ---- Unit Tests ----
option(BUILD_TESTS "Build engine unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
