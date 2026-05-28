#include <gtest/gtest.h>
#include "engine/Logging.hpp"
#include <filesystem>
#include <cstdio>

int main(int argc, char** argv) {
    std::filesystem::create_directories("test_shader_cache");
    te::set_log_sink([](const char* level, const std::string& msg) {
        std::fprintf(stderr, "%s%s\n", level, msg.c_str());
    });
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
