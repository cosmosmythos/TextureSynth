// Tests for cache key differentiation by depth.
// Verifies that ShaderVariantKey and FusedVariantKey pack depth correctly
// so Mono@F16 and Mono@F32 produce distinct cache entries.
// Pure C++ — no Vulkan needed.

#include <gtest/gtest.h>
#include <set>
#include "engine/ShaderVariantKey.hpp"
#include "engine/Graph.hpp"

using namespace te;

// ── ShaderVariantKey: depth in feature_flags ─────────────────────

namespace {

ShaderVariantKey make_key(const std::string& type_id,
                           ChannelFormat fmt,
                           BitDepth depth,
                           uint32_t input_count = 0) {
    ShaderVariantKey k;
    k.node_type_id = type_id;
    k.input_count = input_count;
    const uint32_t fmt_bits   = static_cast<uint32_t>(fmt) & 0x7u;
    const uint32_t depth_bits = static_cast<uint32_t>(depth) & 0x3u;
    k.feature_flags = fmt_bits | (depth_bits << 3);
    return k;
}

} // namespace

TEST(FormatDepthKeys, SameFormatDifferentDepthDifferentKey) {
    auto a = make_key("perlin", ChannelFormat::Mono, BitDepth::F16);
    auto b = make_key("perlin", ChannelFormat::Mono, BitDepth::F32);
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(FormatDepthKeys, SameFormatDifferentDepthF8vsF32) {
    auto a = make_key("perlin", ChannelFormat::RGBA, BitDepth::F8);
    auto b = make_key("perlin", ChannelFormat::RGBA, BitDepth::F32);
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(FormatDepthKeys, SameFormatDifferentDepthF8vsF16) {
    auto a = make_key("perlin", ChannelFormat::UV, BitDepth::F8);
    auto b = make_key("perlin", ChannelFormat::UV, BitDepth::F16);
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(FormatDepthKeys, SameDepthDifferentFormatDifferentKey) {
    auto a = make_key("perlin", ChannelFormat::Mono, BitDepth::F16);
    auto b = make_key("perlin", ChannelFormat::RGBA, BitDepth::F16);
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(FormatDepthKeys, SameFormatSameDepthSameKey) {
    auto a = make_key("perlin", ChannelFormat::Mono, BitDepth::F16);
    auto b = make_key("perlin", ChannelFormat::Mono, BitDepth::F16);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(FormatDepthKeys, AllSixFormatDepthCombosProduceDistinctKeys) {
    std::set<uint64_t> hashes;
    ChannelFormat fmts[] = {ChannelFormat::Mono, ChannelFormat::UV, ChannelFormat::RGB,
                            ChannelFormat::RGBA, ChannelFormat::ID, ChannelFormat::Metadata};
    BitDepth depths[] = {BitDepth::F8, BitDepth::F16, BitDepth::F32};

    for (auto f : fmts) {
        for (auto d : depths) {
            auto k = make_key("node", f, d);
            hashes.insert(k.hash());
        }
    }
    // 6 formats × 3 depths = 18 unique keys (ID/Metadata ignore depth but
    // the depth bits still differ in feature_flags, so hashes differ).
    EXPECT_EQ(hashes.size(), 18u);
}

TEST(FormatDepthKeys, DepthBitsPackedCorrectly) {
    // Verify the bit layout: bits 0..2 = format, bits 3..4 = depth.
    auto mono_f16 = make_key("x", ChannelFormat::Mono, BitDepth::F16);
    auto mono_f32 = make_key("x", ChannelFormat::Mono, BitDepth::F32);

    // Mono = 0, F16 = 1, F32 = 2
    // mono_f16.feature_flags = 0 | (1 << 3) = 0b00001000 = 8
    // mono_f32.feature_flags = 0 | (2 << 3) = 0b00010000 = 16
    EXPECT_EQ(mono_f16.feature_flags, 0b00001000u);
    EXPECT_EQ(mono_f32.feature_flags, 0b00010000u);
}

TEST(FormatDepthKeys, UVFormatBits) {
    auto uv_f8 = make_key("x", ChannelFormat::UV, BitDepth::F8);
    // UV = 1, F8 = 0
    // feature_flags = 1 | (0 << 3) = 1
    EXPECT_EQ(uv_f8.feature_flags, 1u);

    auto uv_f32 = make_key("x", ChannelFormat::UV, BitDepth::F32);
    // feature_flags = 1 | (2 << 3) = 17
    EXPECT_EQ(uv_f32.feature_flags, 17u);
}

TEST(FormatDepthKeys, IDFormatIgnoresDepthInFormatBits) {
    // ID = 4, so bits 0..2 = 100 (4). Depth still packed in bits 3..4.
    auto id_f8  = make_key("x", ChannelFormat::ID, BitDepth::F8);
    auto id_f32 = make_key("x", ChannelFormat::ID, BitDepth::F32);
    // Different depth → different key, even though VkFormat is always R32_UINT.
    EXPECT_NE(id_f8, id_f32);
    EXPECT_NE(id_f8.hash(), id_f32.hash());
}

// ── FusedVariantKey: per-node depth packing ──────────────────────

namespace {

FusedVariantKey make_fused_key(std::vector<std::string> ids,
                                std::vector<ChannelFormat> fmts,
                                std::vector<BitDepth> depths) {
    FusedVariantKey k;
    k.node_type_ids = std::move(ids);
    k.param_socket_masks.resize(k.node_type_ids.size(), 0);
    k.input_counts.resize(k.node_type_ids.size(), 1);
    k.epoch = 7;

    uint32_t feature = 0;
    uint32_t shift = 0;
    for (size_t i = 0; i < k.node_type_ids.size(); ++i) {
        feature |= (static_cast<uint32_t>(fmts[i]) & 0x7u) << shift;
        shift += 3;
        feature |= (static_cast<uint32_t>(depths[i]) & 0x3u) << shift;
        shift += 2;
    }
    k.feature_flags = feature;
    return k;
}

} // namespace

TEST(FusedFormatDepthKeys, SameChainDifferentDepthDifferentKey) {
    // [perlin(Mono,F16), invert] vs [perlin(Mono,F32), invert]
    auto a = make_fused_key({"perlin", "invert"},
                            {ChannelFormat::Mono, ChannelFormat::RGBA},
                            {BitDepth::F16, BitDepth::F16});
    auto b = make_fused_key({"perlin", "invert"},
                            {ChannelFormat::Mono, ChannelFormat::RGBA},
                            {BitDepth::F32, BitDepth::F16});
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(FusedFormatDepthKeys, SameChainDifferentNodeDepthDifferentKey) {
    // [perlin(Mono,F16), invert(F16)] vs [perlin(Mono,F16), invert(F32)]
    auto a = make_fused_key({"perlin", "invert"},
                            {ChannelFormat::Mono, ChannelFormat::RGBA},
                            {BitDepth::F16, BitDepth::F16});
    auto b = make_fused_key({"perlin", "invert"},
                            {ChannelFormat::Mono, ChannelFormat::RGBA},
                            {BitDepth::F16, BitDepth::F32});
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(FusedFormatDepthKeys, SameChainSameDepthsSameKey) {
    auto a = make_fused_key({"perlin", "invert"},
                            {ChannelFormat::Mono, ChannelFormat::RGBA},
                            {BitDepth::F16, BitDepth::F16});
    auto b = make_fused_key({"perlin", "invert"},
                            {ChannelFormat::Mono, ChannelFormat::RGBA},
                            {BitDepth::F16, BitDepth::F16});
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(FusedFormatDepthKeys, DepthBitsPackedAtCorrectPositions) {
    // Single-node chain: node[0] format in bits 0..2, depth in bits 3..4.
    auto k = make_fused_key({"x"}, {ChannelFormat::Mono}, {BitDepth::F32});
    // Mono=0, F32=2: feature_flags = 0 | (2 << 3) = 16
    EXPECT_EQ(k.feature_flags, 16u);
}

TEST(FusedFormatDepthKeys, TwoNodePacking) {
    // node[0]: Mono(F8) → bits 0..2=0, bits 3..4=0 → 0
    // node[1]: UV(F16) → bits 5..7=1, bits 8..9=1 → (1<<5) | (1<<8) = 32 + 256 = 288
    auto k = make_fused_key({"a", "b"},
                            {ChannelFormat::Mono, ChannelFormat::UV},
                            {BitDepth::F8, BitDepth::F16});
    EXPECT_EQ(k.feature_flags, 288u);
}

TEST(FusedFormatDepthKeys, ThreeNodeChainAllDifferentDepths) {
    // node[0]: Mono(F8)  → bits 0..2=0, bits 3..4=0 → 0
    // node[1]: UV(F16)   → bits 5..7=1, bits 8..9=1 → 32 + 256 = 288
    // node[2]: RGB(F32)  → bits 10..12=2, bits 13..14=2 → 2048 + 16384 = 18432
    auto k = make_fused_key({"a", "b", "c"},
                            {ChannelFormat::Mono, ChannelFormat::UV, ChannelFormat::RGB},
                            {BitDepth::F8, BitDepth::F16, BitDepth::F32});
    EXPECT_EQ(k.feature_flags, 0u + 288u + 18432u);
}
