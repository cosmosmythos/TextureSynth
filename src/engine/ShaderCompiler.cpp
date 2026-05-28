#include "engine/ShaderCompiler.hpp"
#include "engine/Logging.hpp"
#include <shaderc/shaderc.hpp>

namespace te {

// Constructor does NOTHING. No threads. No allocations.
// Safe to call at any time, including during Python extension load.
ShaderCompiler::ShaderCompiler() {}

void ShaderCompiler::ensure_workers() {
    if (workers_started_) return;
    workers_started_ = true;
    unsigned n = std::max(2u, std::thread::hardware_concurrency() / 2);
    for (unsigned i = 0; i < n; ++i)
        workers_.emplace_back([this] { worker_loop(); });
    log_info("ShaderCompiler: started " + std::to_string(n) + " worker threads");
}

ShaderCompiler::~ShaderCompiler() {
    if (!workers_started_) return; // nothing to clean up
    {
        std::lock_guard lk(queue_mu_);
        stop_ = true;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_)
        if (t.joinable()) t.join();
}

void ShaderCompiler::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lk(queue_mu_);
            queue_cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

uint64_t ShaderCompiler::hash_source(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}

std::future<CompileResult> ShaderCompiler::compile_compute_async(std::string glsl, std::string name) {
    // Start workers here, not in constructor — safe because by the time
    // anyone calls compile_compute_async(), Python/Blender is fully loaded.
    ensure_workers();

    auto promise = std::make_shared<std::promise<CompileResult>>();
    auto fut = promise->get_future();

    auto task = [glsl = std::move(glsl), name = std::move(name), promise]() mutable {
        CompileResult r;

        shaderc::Compiler compiler;
        shaderc::CompileOptions options;
        options.SetSourceLanguage(shaderc_source_language_glsl);
        options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        options.SetOptimizationLevel(shaderc_optimization_level_performance);

        auto result = compiler.CompileGlslToSpv(
            glsl, shaderc_compute_shader, name.c_str(), options);

        if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
            r.success   = false;
            r.error_log = result.GetErrorMessage();
        } else {
            r.success = true;
            r.spirv.assign(result.cbegin(), result.cend());
        }
        promise->set_value(std::move(r));
    };

    {
        std::lock_guard lk(queue_mu_);
        tasks_.emplace(std::move(task));
    }
    queue_cv_.notify_one();
    return fut;
}

} // namespace te
