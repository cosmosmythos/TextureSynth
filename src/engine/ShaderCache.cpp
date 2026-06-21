#include "engine/ShaderCache.hpp"
#include "engine/Logging.hpp"
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

namespace te {

ShaderCache::ShaderCache(std::string dir) : dir_(std::move(dir)) {
    // Directory creation is handled by Python in cpp_module.py to bypass MSVC locale locking bugs.
}

std::string ShaderCache::path_for_spv_(uint64_t hash) const {
    std::ostringstream oss;
    oss << std::hex << hash << ".spv";
    std::string filename = oss.str();
    if (dir_.empty()) return filename;
    char last = dir_.back();
    if (last == '/' || last == '\\') return dir_ + filename;
    return dir_ + "/" + filename;
}

std::string ShaderCache::path_for_sidecar_(uint64_t hash) const {
    // Sidecar lives next to the .spv. Suffix ".spv.key.json" flags the pair for human-readable listing.
    std::ostringstream oss;
    oss << std::hex << hash << ".spv.key.json";
    std::string filename = oss.str();
    if (dir_.empty()) return filename;
    char last = dir_.back();
    if (last == '/' || last == '\\') return dir_ + filename;
    return dir_ + "/" + filename;
}

std::optional<std::vector<uint32_t>> ShaderCache::read_spv_(const std::string& p) const {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    auto size = (size_t)f.tellg();
    if (size % 4 != 0) return std::nullopt;
    f.seekg(0);
    std::vector<uint32_t> data(size / 4);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (data.empty() || data[0] != 0x07230203u) {
        log_warn("Shader Cache: Invalid SPIR-V blob ignored");
        return std::nullopt;
    }
    return data;
}

void ShaderCache::write_spv_(const std::string& p, const std::vector<uint32_t>& spirv) const {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) { log_warn("shader cache: write failed"); return; }
    f.write(reinterpret_cast<const char*>(spirv.data()),
            (std::streamsize)(spirv.size() * sizeof(uint32_t)));
}

void ShaderCache::write_sidecar_(uint64_t hash, const FusedVariantKey& key) const {
    nlohmann::json j;
    j["node_type_ids"]          = key.node_type_ids;
    j["param_socket_masks"]     = key.param_socket_masks;
    j["input_counts"]           = key.input_counts;
    j["feature_flags"]          = key.feature_flags;
    j["external_socket_masks"]  = key.external_socket_masks;
    j["epoch"]                  = key.epoch;
    std::ofstream f(path_for_sidecar_(hash), std::ios::trunc);
    if (!f) { log_warn("shader cache: sidecar write failed"); return; }
    f << j.dump();
}

bool ShaderCache::sidecar_matches_(uint64_t hash, const FusedVariantKey& key) const {
    std::ifstream f(path_for_sidecar_(hash));
    if (!f) return false;
    nlohmann::json j;
    try { f >> j; }
    catch (const nlohmann::json::parse_error&) { return false; }
    try {
        FusedVariantKey stored;
        stored.node_type_ids          = j.at("node_type_ids").get<std::vector<std::string>>();
        stored.param_socket_masks     = j.at("param_socket_masks").get<std::vector<uint32_t>>();
        stored.input_counts           = j.at("input_counts").get<std::vector<uint32_t>>();
        stored.feature_flags          = j.at("feature_flags").get<uint32_t>();
        stored.external_socket_masks  = j.at("external_socket_masks").get<std::vector<uint32_t>>();
        stored.epoch                  = j.at("epoch").get<uint64_t>();
        return stored == key;
    } catch (const nlohmann::json::exception&) {
        return false;
    }
}

std::optional<std::vector<uint32_t>> ShaderCache::load(const ShaderVariantKey& key) const {
    return read_spv_(path_for_spv_(key.hash()));
}

void ShaderCache::store(const ShaderVariantKey& key, const std::vector<uint32_t>& spirv) const {
    write_spv_(path_for_spv_(key.hash()), spirv);
}

std::optional<std::vector<uint32_t>> ShaderCache::load(const FusedVariantKey& key) const {
    const uint64_t h = key.hash();
    // Sidecar is the equality re-check. If missing (older cache entry, partial write), treat as miss -- next store creates it.
    if (!sidecar_matches_(h, key)) return std::nullopt;
    return read_spv_(path_for_spv_(h));
}

void ShaderCache::store(const FusedVariantKey& key, const std::vector<uint32_t>& spirv) const {
    const uint64_t h = key.hash();
    write_spv_(path_for_spv_(h), spirv);
    write_sidecar_(h, key);
}

} // namespace te
