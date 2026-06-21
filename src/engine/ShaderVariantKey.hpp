#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <functional>
#include <vector>

namespace te {

// Identifies a unique compiled shader variant.
// Keyed by: (node_type, input_count, param_socket_mask, format_override,
//            specialization constants).
// format_override is packed into the low 3 bits of feature_flags (see below).
// specialization[0..specialization_count-1] are raw uint32 constants
// delivered to the pipeline via VkSpecializationInfo at create time.
//
// Cache key = hash of this struct, NOT of the GLSL source string.
// This makes the cache stable across whitespace/comment changes in
// generated GLSL and enables semantic deduplication.
struct ShaderVariantKey {
    std::string node_type_id;       // "blend", "perlin", …
    uint32_t    input_count    = 0; // affects descriptor layout
    uint32_t    param_socket_mask = 0; // bit i = param[i] is socket-driven
    uint32_t    feature_flags  = 0;    // low 3 bits = ChannelFormat (Mono/UV/RGB/RGBA/ID/Metadata).
                                        // High bits reserved for future compile-time flags.
                                        // See GraphCompiler.cpp build_variant_key for packing.
    // Raw specialization constants (Vulkan VkSpecializationInfo payload).
    // When specialization_count == 0, the install path passes nullptr
    // for pSpecializationInfo (preserves prior behavior for un-specialized
    // nodes). Indices 0..7 are valid; out-of-range is silently ignored.
    std::array<uint32_t, 8> specialization{};
    uint32_t                specialization_count = 0;

    bool operator==(const ShaderVariantKey& o) const noexcept {
        if (node_type_id      != o.node_type_id)      return false;
        if (input_count       != o.input_count)       return false;
        if (param_socket_mask != o.param_socket_mask) return false;
        if (feature_flags     != o.feature_flags)     return false;
        if (specialization_count != o.specialization_count) return false;
        // Only compare up to specialization_count — the rest is garbage.
        for (uint32_t i = 0; i < specialization_count && i < 8; ++i) {
            if (specialization[i] != o.specialization[i]) return false;
        }
        return true;
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
        mix(specialization_count);
        for (uint32_t i = 0; i < specialization_count && i < 8; ++i) {
            mix(specialization[i]);
        }
        mix(uint64_t{4}); // epoch: 1=initial, 2=legacy, 3=format_override,
                          //          4=specialization fields present
        return h;
    }
};

// Identifies a unique compiled fused-chain shader. Ordered over the
// chain's node types; different orders hash to different keys.
//
// NO specialization[] field — per-node spec constants live on
// ShaderVariantKey and OR-fold into feature_flags below. Per-pipeline-
// state and per-dispatch values (sliders, time) are push constants by
// Vulkan industry consensus and must NEVER be on a cache key.
//
// cache key = hash() — but the ShaderCache also writes a .key.json
// sidecar and re-checks equality on load. FNV-1a over a vector of
// strings has a higher collision risk than over a single string, so
// the sidecar is the safety net.
struct FusedVariantKey {
    std::vector<std::string> node_type_ids;       // ordered: head first, tail last
    std::vector<uint32_t>    param_socket_masks;  // per-node, same length as node_type_ids
    std::vector<uint32_t>    input_counts;        // per-node declared inputs
    uint32_t                 feature_flags = 0;   // OR-fold across all chain nodes
                                                 // (low 3 bits = format; high bits reserved)
    std::vector<uint32_t>    external_socket_masks; // per-node bitmask: bit s = 1 means
                                                    // socket s is an external (cross-group)
                                                    // input. Same length as node_type_ids.
                                                    // Encodes WHICH sockets are external,
                                                    // not just HOW MANY — fixes cache
                                                    // collision when same count maps to
                                                    // different sockets.
    uint64_t                 epoch         = 7;   // distinct from ShaderVariantKey::epoch=4
                                                  // so the two hash families can't collide
                                                  // if they ever share a directory.
                                                  // epoch 6: added external_inputs (insufficient)
                                                  // epoch 7: external_inputs → external_socket_masks

    bool operator==(const FusedVariantKey& o) const noexcept {
        return node_type_ids          == o.node_type_ids
            && param_socket_masks     == o.param_socket_masks
            && input_counts           == o.input_counts
            && feature_flags          == o.feature_flags
            && external_socket_masks  == o.external_socket_masks
            && epoch                  == o.epoch;
    }

    uint64_t hash() const noexcept {
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
        for (const auto& s : node_type_ids)          for (char c : s) mix(static_cast<uint8_t>(c));
        for (uint32_t m : param_socket_masks)        mix(m);
        for (uint32_t n : input_counts)              mix(n);
        for (uint32_t m : external_socket_masks)     mix(m);
        mix(feature_flags);
        mix(epoch);
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
template<> struct hash<te::FusedVariantKey> {
    size_t operator()(const te::FusedVariantKey& k) const noexcept {
        return static_cast<size_t>(k.hash());
    }
};
} // namespace std
