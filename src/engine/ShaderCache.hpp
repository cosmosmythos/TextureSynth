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
private:
    std::string dir_;
    std::string path_for(uint64_t hash) const;
};

} // namespace te