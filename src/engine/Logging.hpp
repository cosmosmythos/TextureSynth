#pragma once
#include <functional>
#include <mutex>
#include <string_view>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cstdio>
#endif

namespace te {

using LogSink = std::function<void(const char* level, const std::string& message)>;

inline std::mutex& log_sink_mutex() {
    static std::mutex mu;
    return mu;
}

inline LogSink& log_sink() {
    static LogSink sink;
    return sink;
}

inline void set_log_sink(LogSink sink) {
    std::lock_guard<std::mutex> lock(log_sink_mutex());
    log_sink() = std::move(sink);
}

inline void _platform_log(const char* prefix, std::string_view m) {
#ifdef _WIN32
    std::string s = std::string(prefix) + std::string(m) + "\n";
    OutputDebugStringA(s.c_str());
#else
    std::fprintf(stdout, "%s%.*s\n", prefix, (int)m.size(), m.data());
    std::fflush(stdout);
#endif

    LogSink sink_copy;
    {
        std::lock_guard<std::mutex> lock(log_sink_mutex());
        sink_copy = log_sink();
    }
    if (sink_copy) {
        sink_copy(prefix, std::string(m));
    }
}

inline void log_info(std::string_view m)  { _platform_log("[info]  ", m); }
inline void log_warn(std::string_view m)  { _platform_log("[warn]  ", m); }
inline void log_error(std::string_view m) { _platform_log("[error] ", m); }

} // namespace te
