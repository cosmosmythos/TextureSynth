#pragma once
#include "engine/ShaderVariantKey.hpp"
#include <cstdint>
#include <string>
#include <optional>
#include <vector>

namespace te {

class ShaderCache {
public:
    explicit ShaderCache(std::string dir);
    std::optional<std::vector<uint32_t>> load(const ShaderVariantKey& key) const;
    void store(const ShaderVariantKey& key, const std::vector<uint32_t>& spirv) const;
    // Stage 5: fused-chain cache. Same on-disk format (.spv keyed by hash) plus a .key.json sidecar for equality re-check.
    std::optional<std::vector<uint32_t>> load(const FusedVariantKey& key) const;
    void store(const FusedVariantKey& key, const std::vector<uint32_t>& spirv) const;
private:
    std::string dir_;
    std::string path_for_spv_(uint64_t hash) const;
    std::string path_for_sidecar_(uint64_t hash) const;
    std::optional<std::vector<uint32_t>> read_spv_(const std::string& p) const;
    void write_spv_(const std::string& p, const std::vector<uint32_t>& spirv) const;
    // Sidecar: write full key fields next to .spv; on load re-parse and compare to requested key. Mismatch = miss.
    void write_sidecar_(uint64_t hash, const FusedVariantKey& key) const;
    bool sidecar_matches_(uint64_t hash, const FusedVariantKey& key) const;
};

} // namespace te