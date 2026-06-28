// Regression: adding an unrelated node + link changes bindless slot
// assignments for ALL existing chains, corrupting every node's output.
//
// Root cause: set_graph() clears res_sampled_slot_/res_storage_slot_
// and reassigns from scratch. The assignment ORDER depends on pass
// iteration order, which changes when a new node is added.
#include <gtest/gtest.h>
#include "engine/Engine.hpp"
#include "engine/Graph.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/NodeRegistryLoader.hpp"
#include "test_assets.hpp"
#include <thread>
#include <chrono>
#include <cmath>
#include <unordered_set>

using namespace te;

namespace {

NodeLibrary load_real_lib() {
    NodeLibrary lib;
    std::string err;
    int n = NodeRegistryLoader::load_from_directory(
        lib, find_test_nodes_dir(), find_test_glsl_dir(), &err);
    EXPECT_GT(n, 0) << "failed to load real nodes: " << err;
    return lib;
}

bool init_engine(Engine& engine) {
    return engine.init(VK_NULL_HANDLE, nullptr, 0, true, "cache_slot_debug",
                       find_test_nodes_dir().c_str(),
                       find_test_glsl_dir().c_str());
}

bool wait_for_pipeline(Engine& engine, int timeout_ms = 5000) {
    for (int i = 0; i * 10 < timeout_ms; ++i) {
        engine.poll_pending_compiles();
        if (engine.has_pipeline()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return engine.has_pipeline();
}

float mean_r(const std::vector<float>& px) {
    double sum = 0;
    for (size_t i = 0; i + 3 < px.size(); i += 4) sum += px[i];
    return (float)(sum / (px.size() / 4));
}

// Pixel-by-pixel comparison: returns count of pixels that differ.
size_t count_different(const std::vector<float>& a, const std::vector<float>& b) {
    size_t diff = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); i += 4) {
        if (std::abs(a[i] - b[i]) > 0.001f ||
            std::abs(a[i+1] - b[i+1]) > 0.001f ||
            std::abs(a[i+2] - b[i+2]) > 0.001f ||
            std::abs(a[i+3] - b[i+3]) > 0.001f) {
            ++diff;
        }
    }
    return diff;
}

// Build the addon-exact "pre-link" graph: Blend.002 EXISTS but is ISOLATED (no connections).
// The addon always sends ALL tree nodes to the engine, even unreachable ones.
// This matches the addon's 13-node tree with Blend.002 having no links to/from it.
Graph build_addon_pre_graph() {
    Graph g;
    g.nodes.push_back({1,  "worley"});
    g.nodes.push_back({2,  "levels"});
    g.nodes.push_back({3,  "blur"});
    g.nodes.push_back({4,  "invert"});
    g.nodes.push_back({5,  "invert"});
    g.nodes.push_back({6,  "blur"});
    g.nodes.push_back({7,  "perlin"});
    g.nodes.push_back({8,  "levels"});
    g.nodes.push_back({9,  "blend"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blur"});
    g.nodes.push_back({12, "blend"});
    g.nodes.push_back({13, "blend"});  // Blend.002 always exists in tree

    g.connections.push_back({1, 0, 2, 0});   // Worley→Levels
    g.connections.push_back({2, 0, 3, 0});   // Levels→Blur
    g.connections.push_back({2, 0, 5, 1});   // Levels→Invert.001.Color
    g.connections.push_back({3, 0, 4, 1});   // Blur→Invert.Color
    g.connections.push_back({4, 0, 9, 1});   // Invert→Blend.A
    g.connections.push_back({7, 0, 8, 0});   // Perlin→Levels.001
    g.connections.push_back({8, 0, 9, 2});   // Levels.001→Blend.B
    g.connections.push_back({9, 0, 10, 0});  // Blend→Levels.002
    g.connections.push_back({10, 0, 11, 0}); // Levels.002→Blur.002
    g.connections.push_back({5, 0, 6, 0});   // Invert.001→Blur.001
    // NO {6,0,12,1} -- Blur.001→Blend.001.A unconnected (addon match)
    g.connections.push_back({11, 0, 12, 2}); // Blur.002→Blend.001.B

    g.output_node = 12;
    return g;
}

// 13-node graph: add Blur.001→Blend.002.A link
Graph build_addon_post_graph() {
    Graph g = build_addon_pre_graph();
    g.connections.push_back({6, 0, 13, 1});  // Blur.001→Blend.002.A
    return g;
}

// Build the 12-node graph (no Blend.002).
Graph build_12_node_graph() {
    Graph g;
    g.nodes.push_back({1,  "worley"});
    g.nodes.push_back({2,  "levels"});
    g.nodes.push_back({3,  "blur"});
    g.nodes.push_back({4,  "invert"});
    g.nodes.push_back({5,  "invert"});
    g.nodes.push_back({6,  "blur"});
    g.nodes.push_back({7,  "perlin"});
    g.nodes.push_back({8,  "levels"});
    g.nodes.push_back({9,  "blend"});
    g.nodes.push_back({10, "levels"});
    g.nodes.push_back({11, "blur"});
    g.nodes.push_back({12, "blend"});

    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({2, 0, 5, 1});
    g.connections.push_back({3, 0, 4, 1});
    g.connections.push_back({4, 0, 9, 1});
    g.connections.push_back({7, 0, 8, 0});
    g.connections.push_back({8, 0, 9, 2});
    g.connections.push_back({9, 0, 10, 0});
    g.connections.push_back({10, 0, 11, 0});
    g.connections.push_back({5, 0, 6, 0});
    g.connections.push_back({6, 0, 12, 1});
    g.connections.push_back({11, 0, 12, 2});

    g.output_node = 12;
    return g;
}

// Build the 13-node graph (with Blend.002 as dead node).
Graph build_13_node_graph() {
    Graph g = build_12_node_graph();
    g.nodes.push_back({13, "blend"});
    g.connections.push_back({6, 0, 13, 1});  // Blur.001 -> Blend.002.A
    return g;
}

void set_all_params(Engine& engine) {
    engine.update_node_params_by_id(1, {8.0f, 3.0f, 2.0f, 0.5f, 1.0f, 0.0f, 81.0f});
    engine.update_node_params_by_id(7, {31.0f, 5.0f, 2.0f, 0.5f, 0.0f, 0.0f});
    engine.update_node_params_by_id(3, {0.1f});
    engine.update_node_params_by_id(6, {0.05f});
    engine.update_node_params_by_id(11, {0.03f});
    engine.update_node_params_by_id(4, {1.0f});
    engine.update_node_params_by_id(5, {1.0f});
    engine.update_node_params_by_id(9, {1.0f, 0.4533f});
    engine.update_node_params_by_id(12, {0.0f, 0.5f});
    std::vector<float> levels_p = {0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f,
        0.42f, 0.61f, 1.0f, 0.0f, 0.27f, 0.5f, 0.61f, 0.0f, 1.0f,
        0.27f, 0.5f, 0.61f, 0.0f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(2, levels_p);
    std::vector<float> levels001_p = {0.42f, 0.3267f, 0.34f, 0.0f, 0.31f,
        0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.42f, 0.3267f, 0.34f, 0.0f,
        0.31f, 0.42f, 0.3267f, 0.34f, 0.0f, 0.31f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(8, levels001_p);
    std::vector<float> levels002_p = {0.162f, 0.5067f, 0.76f, 0.0f, 0.49f,
        0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.162f, 0.5067f, 0.76f, 0.0f,
        0.49f, 0.162f, 0.5067f, 0.76f, 0.0f, 0.49f, 0.0f, 0.5f, 1.0f,
        0.0f, 1.0f, 0.0f};
    engine.update_node_params_by_id(10, levels002_p);
}

} // namespace


// Test: adding Blend.002 + link MUST NOT change Levels.002 output.
// Levels.002 (node 10) is completely unrelated to Blend.002 (node 13).
// If its output changes, bindless slot assignments are being corrupted
// by the topology change.
TEST(SlotDebug, UnrelatedNode_DoesNotCorrupt_Levels002) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    // Step 1: 12-node graph, read Levels.002
    Graph g1 = build_12_node_graph();
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_NE(gen1, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);

    // Set active node to Levels.002 (node 10)
    uint64_t gen_a = engine.set_active_node(10);
    ASSERT_NE(gen_a, 0u) << engine.last_error();
    auto img1 = engine.readback_sync();
    ASSERT_FALSE(img1.pixels.empty());
    float r1 = mean_r(img1.pixels);
    std::cout << "Step 1 (12-node): Levels.002 R=" << r1 << std::endl;

    // Step 2: 13-node graph (with Blend.002 + Blur.001→Blend.002.A link)
    Graph g2 = build_13_node_graph();
    uint64_t gen2 = engine.set_graph(g2);
    ASSERT_NE(gen2, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);

    // Read Levels.002 again
    gen_a = engine.set_active_node(10);
    ASSERT_NE(gen_a, 0u) << engine.last_error();
    auto img2 = engine.readback_sync();
    ASSERT_FALSE(img2.pixels.empty());
    float r2 = mean_r(img2.pixels);
    std::cout << "Step 2 (13-node): Levels.002 R=" << r2 << std::endl;

    size_t diff = count_different(img1.pixels, img2.pixels);
    std::cout << "  pixel diff: " << diff << " / " << (img1.pixels.size()/4) << std::endl;

    // Levels.002 must produce identical output — it has zero connection to Blend.002
    EXPECT_NEAR(r2, r1, 0.001f)
        << "BUG: Adding Blend.002 + link changed Levels.002 output!"
        << " Expected R=" << r1 << " got R=" << r2;
    EXPECT_EQ(diff, 0u)
        << "BUG: " << diff << " pixels changed in Levels.002"
        << " after adding unrelated Blend.002 + link!";
}


// Test: adding Blend.002 + link MUST NOT change Blend.001 output
// when Blend.001.A is connected (full graph).
TEST(SlotDebug, UnrelatedNode_DoesNotCorrupt_Blend001) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    // Step 1: 12-node graph, read Blend.001
    Graph g1 = build_12_node_graph();
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_NE(gen1, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 0.5f});

    auto img1 = engine.readback_sync();
    ASSERT_FALSE(img1.pixels.empty());
    float r1 = mean_r(img1.pixels);
    std::cout << "Step 1 (12-node): Blend.001 R=" << r1 << std::endl;

    // Step 2: 13-node graph
    Graph g2 = build_13_node_graph();
    uint64_t gen2 = engine.set_graph(g2);
    ASSERT_NE(gen2, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);
    engine.update_node_params_by_id(12, {0.0f, 0.5f});

    auto img2 = engine.readback_sync();
    ASSERT_FALSE(img2.pixels.empty());
    float r2 = mean_r(img2.pixels);
    std::cout << "Step 2 (13-node): Blend.001 R=" << r2 << std::endl;

    size_t diff = count_different(img1.pixels, img2.pixels);
    std::cout << "  pixel diff: " << diff << " / " << (img1.pixels.size()/4) << std::endl;

    EXPECT_NEAR(r2, r1, 0.01f)
        << "BUG: Adding Blend.002 + link changed Blend.001 output!"
        << " Expected R=" << r1 << " got R=" << r2;
    EXPECT_EQ(diff, 0u)
        << "BUG: " << diff << " pixels changed in Blend.001"
        << " after adding unrelated Blend.002 + link!";
}


// Test: addon-exact topology — Blend.002 EXISTS as isolated node in BOTH steps.
// The addon always sends ALL tree nodes. Only the link changes.
TEST(SlotDebug, AddonExact_IsolatedBlend002_ThenConnected) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    // Step 1: addon-exact pre-link graph (Blend.002 isolated, no connections)
    Graph g1 = build_addon_pre_graph();
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_NE(gen1, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);
    engine.update_node_params_by_id(3, {0.05f});
    engine.update_node_params_by_id(12, {0.0f, 0.0f});
    engine.update_node_params_by_id(13, {0.0f, 1.0f});  // Blend.002 params (always pushed)

    auto img1 = engine.readback_sync();
    ASSERT_FALSE(img1.pixels.empty());
    float r1 = mean_r(img1.pixels);
    std::cout << "Step 1 (pre-link, 13 nodes, Blend.002 isolated): Blend.001 R=" << r1 << std::endl;

    // Step 2: add link Blur.001→Blend.002.A (same 13 nodes, one extra connection)
    Graph g2 = build_addon_post_graph();
    uint64_t gen2 = engine.set_graph(g2);
    ASSERT_NE(gen2, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);
    engine.update_node_params_by_id(3, {0.05f});
    engine.update_node_params_by_id(12, {0.0f, 0.0f});
    engine.update_node_params_by_id(13, {0.0f, 1.0f});  // Blend.002 params (same)

    auto img2 = engine.readback_sync();
    ASSERT_FALSE(img2.pixels.empty());
    float r2 = mean_r(img2.pixels);

    size_t diff = count_different(img1.pixels, img2.pixels);
    std::cout << "Step 2 (post-link, one new connection): Blend.001 R=" << r2
              << " diff=" << diff << "/" << (img1.pixels.size()/4) << std::endl;

    EXPECT_NEAR(r2, r1, 0.001f)
        << "BUG: Adding Blur.001→Blend.002.A changed Blend.001!"
        << " R1=" << r1 << " R2=" << r2;
    EXPECT_EQ(diff, 0u)
        << "BUG: " << diff << " pixels changed after adding link";
}


// Test: every node's output must be identical between the two topologies.
// Uses set_active_node to read each intermediate.
TEST(SlotDebug, AllNodes_Unchanged_AfterUnrelatedLink) {
    Engine engine;
    ASSERT_TRUE(init_engine(engine));

    // Step 1: 12-node graph
    Graph g1 = build_12_node_graph();
    uint64_t gen1 = engine.set_graph(g1);
    ASSERT_NE(gen1, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);

    // Read Levels (2), Levels.001 (8), Levels.002 (10)
    std::unordered_map<NodeId, float> means1;
    for (NodeId nid : {2, 8, 10, 11, 12}) {
        engine.set_active_node(nid);
        auto img = engine.readback_sync();
        if (!img.pixels.empty()) {
            means1[nid] = mean_r(img.pixels);
            std::cout << "Step 1 node " << nid << " R=" << means1[nid] << std::endl;
        }
    }

    // Step 2: 13-node graph
    Graph g2 = build_13_node_graph();
    uint64_t gen2 = engine.set_graph(g2);
    ASSERT_NE(gen2, 0u) << engine.last_error();
    ASSERT_TRUE(wait_for_pipeline(engine));

    set_all_params(engine);

    // Read same nodes
    std::unordered_map<NodeId, float> means2;
    for (NodeId nid : {2, 8, 10, 11, 12}) {
        engine.set_active_node(nid);
        auto img = engine.readback_sync();
        if (!img.pixels.empty()) {
            means2[nid] = mean_r(img.pixels);
            std::cout << "Step 2 node " << nid << " R=" << means2[nid] << std::endl;
        }
    }

    // Compare
    for (NodeId nid : {2, 8, 10, 11, 12}) {
        if (means1.count(nid) && means2.count(nid)) {
            float d = std::abs(means2[nid] - means1[nid]);
            std::cout << "  node " << nid << " delta=" << d << std::endl;
            EXPECT_LE(d, 0.001f)
                << "BUG: node " << nid << " changed after adding unrelated Blend.002!"
                << " Expected R=" << means1[nid] << " got R=" << means2[nid];
        }
    }
}
