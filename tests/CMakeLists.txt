include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(engine_tests
    main.cpp
    test_context.cpp
    test_graph_validation.cpp
    test_resource_manager.cpp
    test_async_readback.cpp
    test_full_pipeline.cpp
    test_shader_variant_key.cpp
    test_fused_variant_key.cpp
    test_dirty_set.cpp
    test_chain_finding.cpp
    test_chain_shader_emitter.cpp
)

target_link_libraries(engine_tests PRIVATE
    engine
    GTest::gtest
    GTest::gmock
)

target_include_directories(engine_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Force validation layers on for tests
target_compile_definitions(engine_tests PRIVATE TS_TESTS_ENABLE_VALIDATION=1)

# Hard-code the project root at build time so test binaries can find
# shader_assets regardless of the working directory they were launched
# from. See tests/test_assets.hpp for the resolution order (env var
# override > this macro > walk-up from CWD > CWD-relative fallback).
target_compile_definitions(engine_tests PRIVATE
    TEXTURESYNTH_TEST_ASSET_DIR="${CMAKE_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(engine_tests)
