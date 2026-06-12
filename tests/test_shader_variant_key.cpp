// Layer 1: Direct ShaderVariantKey unit tests — no engine/Vulkan needed.

#include <gtest/gtest.h>
#include <set>
#include <string>
#include "engine/ShaderVariantKey.hpp"

using namespace te;

TEST(ShaderVariantKey, DefaultKeyHasAllFieldsZero) {
    ShaderVariantKey k;
    EXPECT_EQ(k.node_type_id, "");
    EXPECT_EQ(k.input_count,       0u);
    EXPECT_EQ(k.param_socket_mask, 0u);
    EXPECT_EQ(k.feature_flags,     0u);
    EXPECT_EQ(k.specialization_count, 0u);
    for (auto v : k.specialization) EXPECT_EQ(v, 0u);
}

TEST(ShaderVariantKey, HashIsDeterministic) {
    ShaderVariantKey a;
    a.node_type_id = "perlin";
    a.input_count  = 0;
    a.feature_flags = 0u;
    ShaderVariantKey b = a;
    EXPECT_EQ(a.hash(), b.hash());
    EXPECT_EQ(a, b);
}

TEST(ShaderVariantKey, HashChangesWithFeatureFlags) {
    ShaderVariantKey mono;
    mono.node_type_id = "perlin";
    mono.feature_flags = 0u;

    ShaderVariantKey uv;
    uv.node_type_id = "perlin";
    uv.feature_flags = 1u;

    EXPECT_NE(mono.hash(), uv.hash());
    EXPECT_NE(mono, uv);
}

TEST(ShaderVariantKey, HashChangesWithSpecialization) {
    ShaderVariantKey a;
    a.node_type_id = "x";
    a.specialization_count = 1;
    a.specialization[0] = 7;

    ShaderVariantKey b;
    b.node_type_id = "x";
    b.specialization_count = 1;
    b.specialization[0] = 9;

    EXPECT_NE(a.hash(), b.hash());
    EXPECT_NE(a, b);
}

TEST(ShaderVariantKey, SpecializationTailDoesNotAffectHashWhenCountIsZero) {
    ShaderVariantKey a;
    a.node_type_id = "x";
    a.specialization_count = 0;
    a.specialization[5] = 0xDEADBEEF;

    ShaderVariantKey b;
    b.node_type_id = "x";
    b.specialization_count = 0;
    b.specialization[5] = 0xCAFEBABE;

    EXPECT_EQ(a.hash(), b.hash());
    EXPECT_EQ(a, b);
}

TEST(ShaderVariantKey, EqualityIsReflexiveSymmetricTransitive) {
    ShaderVariantKey base;
    base.node_type_id = "perlin";
    base.input_count  = 0;
    base.feature_flags = 2u;
    base.specialization_count = 2;
    base.specialization[0] = 1;
    base.specialization[1] = 2;

    ShaderVariantKey copy = base;
    EXPECT_EQ(base, copy);
    EXPECT_EQ(base.hash(), copy.hash());

    copy.specialization[1] = 3;
    EXPECT_NE(base, copy);
}

TEST(ShaderVariantKey, StdHashIsConsistentWithOperatorEq) {
    std::hash<ShaderVariantKey> hasher;
    ShaderVariantKey a;
    a.node_type_id = "perlin";
    a.specialization_count = 1;
    a.specialization[0] = 42;
    ShaderVariantKey b = a;
    EXPECT_EQ(hasher(a), hasher(b));
}

TEST(ShaderVariantKey, EpochIsMixedIn) {
    ShaderVariantKey k;
    EXPECT_NE(k.hash(), 1469598103934665603ull);
}

TEST(ShaderVariantKey, EightDistinctSpecializationValuesProduceEightDistinctHashes) {
    std::set<uint64_t> hashes;
    for (uint32_t i = 0; i < 8; ++i) {
        ShaderVariantKey k;
        k.node_type_id = "x";
        k.specialization_count = static_cast<uint32_t>(i) + 1;
        k.specialization[i]    = 0xC0FFEEu + i;
        hashes.insert(k.hash());
    }
    EXPECT_EQ(hashes.size(), 8u);
}

TEST(ShaderVariantKey, SpecializationCountZeroProducesIdenticalHash) {
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 0;
    a.specialization.fill(0xDEADBEEF);
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 0;
    b.specialization.fill(0);

    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}

TEST(ShaderVariantKey, SpecializationCountIsPartOfTheHash) {
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 0;
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 1;
    b.specialization[0] = 0;
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(ShaderVariantKey, SpecializationSlotsBeyondIndexSevenAreIgnored) {
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 8;
    a.specialization[7] = 1;
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 8;
    b.specialization[7] = 2;
    EXPECT_NE(a, b);
    EXPECT_NE(a.hash(), b.hash());
}

TEST(ShaderVariantKey, SpecializationCountOneOnlyAffectsSlotZero) {
    ShaderVariantKey a; a.node_type_id = "blend";
    a.specialization_count = 1;
    a.specialization[0] = 42;
    a.specialization[1] = 99;
    ShaderVariantKey b; b.node_type_id = "blend";
    b.specialization_count = 1;
    b.specialization[0] = 42;
    b.specialization[1] = 0;

    EXPECT_EQ(a, b);
    EXPECT_EQ(a.hash(), b.hash());
}
