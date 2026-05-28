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
)

target_link_libraries(engine_tests PRIVATE
    engine
    GTest::gtest
    GTest::gmock
)

target_include_directories(engine_tests PRIVATE ${CMAKE_SOURCE_DIR}/src)

# Force validation layers on for tests
target_compile_definitions(engine_tests PRIVATE TS_TESTS_ENABLE_VALIDATION=1)

include(GoogleTest)
gtest_discover_tests(engine_tests)
