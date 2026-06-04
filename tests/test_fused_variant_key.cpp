// Stage 5.1 tests: FusedVariantKey invariants.
// All pure C++ — no Vulkan, no shaderc, no disk I/O. Run in <1ms.

#include <gtest/gtest.h>
#include <set>
#include "engine/ShaderVariantKey.hpp"

using namespace te;

namespace {

FusedVariantKey make_chain(std::vector<std::string> ids,
                           std::vector<uint32_t> masks = {},
                           std::vector<uint32_t> ins   = {}) {
    FusedVariantKey k;
    k.node_type_ids      = std::move(ids);
    k.param_socket_masks = std::move(masks);
    k.input_counts       = std::move(ins);
    return k;
}

} // namespace

// Order matters: [perlin, invert, grayscale] != [grayscale, invert, perlin].
// This is the entire point of a FusedVariantKey over a per-node key.
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
    a.epoch = 5;
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
    // Std hash specialization agrees with hash().
    std::hash<FusedVariantKey> hasher;
    EXPECT_EQ(hasher(k), static_cast<size_t>(h1));
}

// Epoch distinctness: FusedVariantKey::epoch=5 differs from
// ShaderVariantKey::epoch=4. If these two hashes ever shared a
// directory (unlikely today, possible in a future consolidation),
// the epoch makes them occupy disjoint hash namespaces.
TEST(FusedVariantKey, EpochDistinctFromPerNode) {
    // Build a FusedVariantKey and a ShaderVariantKey with the same
    // string-id "perlin", zero everywhere else, and check that the
    // resulting hashes are different.
    FusedVariantKey fk;
    fk.node_type_ids = {"perlin"};
    fk.param_socket_masks = {0};
    fk.input_counts = {0};
    fk.epoch = 5;

    ShaderVariantKey sk;
    sk.node_type_id = "perlin";
    sk.input_count = 0;
    // ShaderVariantKey::epoch = 4 is mixed into sk.hash().

    // FNV-1a mixes the epoch at the END for both keys; if they used
    // the same epoch they'd risk identical hashes for the same input.
    // We don't assert the actual hashes (that would be brittle); we
    // assert that the per-node key with a single perlin input doesn't
    // collide with the fused key for a one-node chain.
    EXPECT_NE(fk.hash(), static_cast<uint64_t>(std::hash<ShaderVariantKey>{}(sk)));
}
