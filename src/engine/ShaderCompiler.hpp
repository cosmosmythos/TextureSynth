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
    ShaderVariantKey variant_key;
};

class ShaderCompiler {
public:
    ShaderCompiler();
    ~ShaderCompiler();

    std::future<CompileResult> compile_compute_async(std::string glsl, std::string name);

    // Sync wrapper. Chain compile is short (small SPIR-V) and runs at set_graph time; thread-pool hop not worth the cost.
    CompileResult compile_compute_sync(std::string glsl, std::string name) {
        return compile_compute_async(std::move(glsl), std::move(name)).get();
    }

    static uint64_t hash_source(const std::string& s);

private:
    void worker_loop();

    // NOT started in constructor. Called on first compile_compute_async() to avoid CRT TLS issues during Python extension load.
    void ensure_workers();

    std::vector<std::thread> workers_;
    std::mutex               queue_mu_;
    std::condition_variable  queue_cv_;
    std::queue<std::function<void()>> tasks_;
    std::atomic<bool>        stop_{false};
    std::atomic<bool>        workers_started_{false};
};

} // namespace te
