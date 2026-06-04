// Stage 6.1 tests: DirtySet::is_chain_dirty + set_chain_membership.

#include <gtest/gtest.h>
#include "engine/Graph.hpp"      // NodeId
#include "engine/DirtySet.hpp"

using te::DirtySet;

TEST(DirtySet, IsChainDirtyWhenAnyMemberDirty) {
    DirtySet ds; ds.clear();
    ds.set_chain_membership({{0, {10,11,12}}, {1, {20,21}}});
    ds.mark_node(11);
    EXPECT_TRUE(ds.is_chain_dirty(0));
    EXPECT_FALSE(ds.is_chain_dirty(1));
}

TEST(DirtySet, IsChainCleanWhenAllMembersClean) {
    DirtySet ds; ds.clear();
    ds.set_chain_membership({{0, {10,11,12}}});
    EXPECT_FALSE(ds.is_chain_dirty(0));
}

TEST(DirtySet, MarkTopologyChangeMakesAllChainsDirty) {
    DirtySet ds; ds.clear();
    ds.set_chain_membership({{0, {10,11}}});
    EXPECT_FALSE(ds.is_chain_dirty(0));
    ds.mark_topology_change();
    EXPECT_TRUE(ds.is_chain_dirty(0));
}

TEST(DirtySet, CacheInvalidatesOnMarkNode) {
    DirtySet ds; ds.clear();
    ds.set_chain_membership({{0, {10,11}}});
    EXPECT_FALSE(ds.is_chain_dirty(0));   // caches false
    ds.mark_node(11);
    EXPECT_TRUE(ds.is_chain_dirty(0));    // cache invalidated
}
