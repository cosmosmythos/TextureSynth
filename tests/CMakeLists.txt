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
    test_fusion_group.cpp
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
