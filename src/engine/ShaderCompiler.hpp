#pragma once
#include "engine/ShaderVariantKey.hpp"
#include <cstdint>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <functional>

namespace te {

struct CompileResult {
    bool success = false;
    std::vector<uint32_t> spirv;
    std::string error_log;
    ShaderVariantKey variant_key;  // replaces source_hash
};

class ShaderCompiler {
public:
    ShaderCompiler();
    ~ShaderCompiler();

    // Submit a GLSL compute shader to be compiled off the main thread.
    std::future<CompileResult> compile_compute_async(std::string glsl, std::string name);

    static uint64_t hash_source(const std::string& s);

private:
    void worker_loop();

    // Threads are NOT started in the constructor.
    // Called on first compile_compute_async() to avoid CRT TLS issues
    // when the compiler is constructed during Python extension load.
    void ensure_workers();

    std::vector<std::thread> workers_;
    std::mutex               queue_mu_;
    std::condition_variable  queue_cv_;
    std::queue<std::function<void()>> tasks_;
    std::atomic<bool>        stop_{false};
    bool                     workers_started_{false};
};

} // namespace te
