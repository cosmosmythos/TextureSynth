#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/DirtySet.hpp"

using te::DirtySet;

TEST(DirtySet, MarkNodeMakesDirty) {
    DirtySet ds; ds.clear();
    ds.mark_node(11);
    EXPECT_TRUE(ds.is_dirty(11));
    EXPECT_FALSE(ds.is_dirty(99));
}

TEST(DirtySet, TopologyChangeMakesAllDirty) {
    DirtySet ds; ds.clear();
    ds.mark_topology_change();
    EXPECT_TRUE(ds.any());
}

TEST(DirtySet, ClearResetsState) {
    DirtySet ds; ds.clear();
    ds.mark_node(1);
    EXPECT_TRUE(ds.any());
    ds.clear();
    EXPECT_FALSE(ds.any());
}

TEST(DirtySet, PropagateExpandsDownstream) {
    DirtySet ds; ds.clear();
    std::unordered_map<uint64_t, std::vector<uint64_t>> adj;
    adj[10] = {20, 30};
    adj[20] = {40};
    ds.mark_node(10);
    ds.propagate(adj);
    EXPECT_TRUE(ds.is_dirty(10));
    EXPECT_TRUE(ds.is_dirty(20));
    EXPECT_TRUE(ds.is_dirty(30));
    EXPECT_TRUE(ds.is_dirty(40));
    EXPECT_FALSE(ds.is_dirty(99));
}
