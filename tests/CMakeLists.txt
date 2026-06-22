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
    test_async_readback.cpp
    test_full_pipeline.cpp
    test_shader_variant_key.cpp
    test_fused_variant_key.cpp
    test_dirty_set.cpp
    test_aliasing.cpp
    test_mask_node.cpp
    test_timestamps.cpp
    inspect_perlin_invert.cpp
    test_active_path_tracer.cpp
    test_fused_graph_emitter.cpp
    test_fused_graph_compiler.cpp
    test_fused_graph_activation.cpp
    test_fused_real_nodes.cpp
    test_combine_rgba.cpp
    test_storage_format.cpp
    test_blend_mode_param.cpp
    test_depth_resolution.cpp
    test_format_depth_keys.cpp
    test_dump_gabor_blend.cpp
    test_repro_blend_preview.cpp
    test_research_register_pressure.cpp
    test_image_upload.cpp
    test_noise_nodes.cpp
    test_simplex_debug.cpp
)

target_link_libraries(engine_tests PRIVATE
    engine
    GTest::gtest
    GTest::gmock
)

target_include_directories(engine_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_compile_definitions(engine_tests PRIVATE TS_TESTS_ENABLE_VALIDATION=1)
target_compile_definitions(engine_tests PRIVATE
    TEXTURESYNTH_TEST_ASSET_DIR="${CMAKE_SOURCE_DIR}")

include(GoogleTest)
gtest_discover_tests(engine_tests)
