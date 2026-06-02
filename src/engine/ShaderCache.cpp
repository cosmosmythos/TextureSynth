#include "engine/ShaderCache.hpp"
#include "engine/Logging.hpp"
#include <fstream>
#include <sstream>

namespace te {

ShaderCache::ShaderCache(std::string dir) : dir_(std::move(dir)) {
    // Directory creation is handled safely by Python in cpp_module.py
    // to bypass MSVC C++ standard library locale locking bugs.
}

std::string ShaderCache::path_for(uint64_t hash) const {
    std::ostringstream oss;
    oss << std::hex << hash << ".spv";
    std::string filename = oss.str();
    
    // Simple path join to avoid std::filesystem
    if (dir_.empty()) return filename;
    char last = dir_.back();
    if (last == '/' || last == '\\') return dir_ + filename;
    return dir_ + "/" + filename;
}

std::optional<std::vector<uint32_t>> ShaderCache::load(const ShaderVariantKey& key) const {
    auto p = path_for(key.hash());
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    auto size = (size_t)f.tellg();
    if (size % 4 != 0) return std::nullopt;
    f.seekg(0);
    std::vector<uint32_t> data(size / 4);
    // Validate the SPIR-V magic number. A corrupt or stale cache file
    // won't start with 0x07230203 — let the caller fall through to
    // recompile rather than passing garbage to vkCreateShaderModule.
    if (data.empty() || data[0] != 0x07230203u) {
        log_warn("Shader Cache: Invalid SPIR-V blob ignored");
        return std::nullopt;
    }
    return data;
}

void ShaderCache::store(const ShaderVariantKey& key, const std::vector<uint32_t>& spirv) const {
    auto p = path_for(key.hash());
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) { log_warn("shader cache: write failed"); return; }
    f.write(reinterpret_cast<const char*>(spirv.data()),
            (std::streamsize)(spirv.size() * sizeof(uint32_t)));
}

} // namespace te