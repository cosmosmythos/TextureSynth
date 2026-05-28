#pragma once
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>
#include <cstring>
#include <source_location>
#include <stdexcept>
#include <string>
#include "Logging.hpp"

namespace te::detail {

inline const char* short_fn(const char* path) {
    if (const char* bs = std::strrchr(path, '\\')) path = bs + 1;
    if (const char* fs = std::strrchr(path, '/'))  path = fs + 1;
    return path;
}

inline VkResult check_vk(VkResult res, const char* expr,
                         const std::source_location& loc) {
    if (res != VK_SUCCESS) {
        std::string msg = std::string(expr) + " failed: "
                        + string_VkResult(res) + " ("
                        + short_fn(loc.file_name()) + ":"
                        + std::to_string(loc.line()) + ", "
                        + loc.function_name() + ")";
        log_error(msg);
    }
    return res;
}

} // namespace te::detail

#define VK_CHECK(expr) \
    ::te::detail::check_vk((expr), #expr, std::source_location::current())

#define VK_CHECK_THROW(expr)                                                      \
    do {                                                                          \
        VkResult _vk_res = (expr);                                                \
        if (_vk_res != VK_SUCCESS) {                                              \
            auto _vk_loc = std::source_location::current();                       \
            ::te::detail::check_vk(_vk_res, #expr, _vk_loc);                     \
            throw std::runtime_error(                                             \
                std::string(#expr) + " failed: " + string_VkResult(_vk_res)       \
                + " (" + ::te::detail::short_fn(_vk_loc.file_name()) + ":"                      \
                + std::to_string(_vk_loc.line()) + ")");                          \
        }                                                                         \
    } while (0)
