#pragma once
#include <cstdint>
#include <string>
#include <functional>

namespace te {

// Identifies a unique compiled shader variant.
// Today: keyed by (node_type, input_count, socket_param_mask, output_format).
// Tomorrow: feature_flags drives #define emission.
//
// Cache key = hash of this struct, NOT of the GLSL source string.
// This makes the cache stable across whitespace/comment changes in
// generated GLSL and enables semantic deduplication.
struct ShaderVariantKey {
    std::string node_type_id;       // "blend", "perlin", …
    uint32_t    input_count    = 0; // affects descriptor layout
    uint32_t    param_socket_mask = 0; // bit i = param[i] is socket-driven
    uint32_t    feature_flags  = 0; // reserved: bit 0 = USE_MASK, bit 1 = HQ, …

    bool operator==(const ShaderVariantKey& o) const noexcept {
        return node_type_id      == o.node_type_id
            && input_count       == o.input_count
            && param_socket_mask == o.param_socket_mask
            && feature_flags     == o.feature_flags;
    }

    uint64_t hash() const noexcept {
        // FNV-1a over all fields.
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) {
            h ^= v; h *= 1099511628211ull;
        };
        for (char c : node_type_id) mix(static_cast<uint8_t>(c));
        mix(input_count);
        mix(param_socket_mask);
        mix(feature_flags);
        mix(uint64_t{2}); // epoch: increment when binding layout changes
        return h;
    }
};

} // namespace te

namespace std {
template<> struct hash<te::ShaderVariantKey> {
    size_t operator()(const te::ShaderVariantKey& k) const noexcept {
        return static_cast<size_t>(k.hash());
    }
};
} // namespace std
