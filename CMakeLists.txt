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
)
target_include_directories(engine PUBLIC src)
target_link_libraries(engine PUBLIC
    Vulkan::Vulkan
    Vulkan::shaderc_combined
    vk-bootstrap::vk-bootstrap
    GPUOpen::VulkanMemoryAllocator
    nlohmann_json::nlohmann_json
)

# ---- Viewer executable ----
add_executable(viewer
    src/viewer/main.cpp
    src/viewer/Window.cpp
    src/viewer/Swapchain.cpp
    src/viewer/ImGuiLayer.cpp
    src/viewer/Renderer.cpp
)
target_link_libraries(viewer PRIVATE engine glfw imgui)

# Copy shaders next to executable
add_custom_command(TARGET viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shader_assets
        $<TARGET_FILE_DIR:viewer>/shader_assets
)

# ---- Python Extension (TextureSynth Addon) ----
option(BUILD_PYTHON_BINDINGS "Build PyBind11 extension" OFF)
if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6)
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(texturesynth_core src/bindings/bindings.cpp)
    target_link_libraries(texturesynth_core PRIVATE engine)
endif()

# ---- Unit Tests ----
option(BUILD_TESTS "Build engine unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()