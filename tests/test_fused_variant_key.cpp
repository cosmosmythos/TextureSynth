// Stage 5.1 tests: FusedVariantKey invariants.
// All pure C++ — no Vulkan, no shaderc, no disk I/O. Run in <1ms.

#include <gtest/gtest.h>
#include <set>
#include "engine/ShaderVariantKey.hpp"

using namespace te;

namespace {

FusedVariantKey make_chain(std::vector<std::string> ids,
                           std::vector<uint32_t> masks = {},
                           std::vector<uint32_t> ins   = {},
                           std::vector<uint32_t> ipi   = {}) {
    FusedVariantKey k;
    k.node_type_ids             = std::move(ids);
    k.param_socket_masks        = std::move(masks);
    k.input_counts              = std::move(ins);
    k.internal_producer_indices = std::move(ipi);
    return k;
}

} // namespace

// Order matters: [perlin, invert, grayscale] != [grayscale, invert, perlin].
TEST(FusedVariantKey, OrderMatters) {
    auto a = make_chain({"perlin", "invert", "grayscale"}, {0,0,0}, {0,1,1});
    auto b = make_chain({"grayscale", "invert", "perlin"}, {0,0,0}, {0,1,1});
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

// Identical chain → identical key.
TEST(FusedVariantKey, IdenticalChainSameKey) {
    auto a = make_chain({"perlin", "invert"}, {0, 0}, {0, 1});
    a.feature_flags = 0;
    a.epoch = 8;
    auto b = a;
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}

// Different param_socket_mask on the SAME node sequence → different key.
TEST(FusedVariantKey, DifferentParamMaskDifferentKey) {
    auto a = make_chain({"step", "step"}, {0, 0}, {1, 1});
    auto b = make_chain({"step", "step"}, {1, 1}, {1, 1});
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

// Different input_count on the SAME node sequence → different key.
TEST(FusedVariantKey, DifferentInputCountDifferentKey) {
    auto a = make_chain({"blend", "step"}, {0, 0}, {2, 1});
    auto b = make_chain({"blend", "step"}, {0, 0}, {1, 1});
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

// Hash is deterministic: same struct, same hash, run after run.
TEST(FusedVariantKey, HashDeterministic) {
    auto k = make_chain({"perlin", "invert", "grayscale"}, {0,0,0}, {0,1,1});
    const uint64_t h1 = k.hash();
    const uint64_t h2 = k.hash();
    EXPECT_EQ(h1, h2);
    std::hash<FusedVariantKey> hasher;
    EXPECT_EQ(hasher(k), static_cast<size_t>(h1));
}

// Epoch distinctness: FusedVariantKey::epoch=8 differs from
// ShaderVariantKey::epoch=5.
TEST(FusedVariantKey, EpochDistinctFromPerNode) {
    FusedVariantKey fk;
    fk.node_type_ids = {"perlin"};
    fk.param_socket_masks = {0};
    fk.input_counts = {0};
    fk.epoch = 8;

    ShaderVariantKey sk;
    sk.node_type_id = "perlin";
    sk.input_count = 0;

    EXPECT_NE(fk.hash(), static_cast<uint64_t>(std::hash<ShaderVariantKey>{}(sk)));
}

// Different external_socket_mask on the SAME node sequence → different key.
TEST(FusedVariantKey, DifferentSocketMaskDifferentKey) {
    auto a = make_chain({"worley", "levels", "blend"}, {0, 0, 0}, {0, 1, 3});
    a.external_socket_masks = {0, 0, 0};
    a.epoch = 8;
    auto b = make_chain({"worley", "levels", "blend"}, {0, 0, 0}, {0, 1, 3});
    b.external_socket_masks = {0, 0, 0b100}; // blend socket 2
    b.epoch = 8;
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

// Same count, different socket → different key (the real fix).
// Two chains with [worley, levels, blend] where both have 1 external
// input but on different sockets must produce different keys.
TEST(FusedVariantKey, SameCountDifferentSocketDifferentKey) {
    auto a = make_chain({"worley", "levels", "blend"}, {0, 0, 0}, {0, 1, 3});
    a.external_socket_masks = {0, 0, 0b100}; // blend socket 2 (B)
    a.epoch = 8;
    auto b = make_chain({"worley", "levels", "blend"}, {0, 0, 0}, {0, 1, 3});
    b.external_socket_masks = {0, 0, 0b010}; // blend socket 1 (A)
    b.epoch = 8;
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

// Same node types + same external masks + SWAPPED internal wiring → different key.
// Reproduces the user-reported Blender symptom: two blend nodes share producer
// types (e.g. [value, simplex, blend]) but assign them to A/B sockets in
// swapped order. Pre-fix, both emitted different GLSL yet collided on every
// other key field, so the cache served the first graph's SPIR-V for the second.
// internal_producer_indices is the per-socket producer local_index that breaks
// the collision.
TEST(FusedVariantKey, DifferentInternalWiringDifferentKey) {
    // Chain layout: [value(0), simplex(1), blend(2)]. blend has 3 inputs
    // (mask, a, b). mask is unconnected (UINT32_MAX); a and b are RegSrc.
    // Graph A: value(0)->a, simplex(1)->b  =>  ipi = [UINT32_MAX, 0, 1]
    // Graph B: simplex(1)->a, value(0)->b  =>  ipi = [UINT32_MAX, 1, 0]
    auto a = make_chain({"value", "simplex", "blend"}, {0, 0, 0}, {0, 0, 3},
                        {UINT32_MAX, 0, 1});
    a.external_socket_masks = {0, 0, 0};
    a.epoch = 8;
    auto b = make_chain({"value", "simplex", "blend"}, {0, 0, 0}, {0, 0, 3},
                        {UINT32_MAX, 1, 0});
    b.external_socket_masks = {0, 0, 0};
    b.epoch = 8;
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());

    // Sanity: identical wiring on both → identical key (guards against a hash
    // that ignores the field entirely and just always differs).
    auto c = make_chain({"value", "simplex", "blend"}, {0, 0, 0}, {0, 0, 3},
                        {UINT32_MAX, 0, 1});
    c.external_socket_masks = {0, 0, 0};
    c.epoch = 8;
    EXPECT_EQ(a, c);
    EXPECT_EQ(a.hash(), c.hash());
}
