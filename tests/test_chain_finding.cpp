#include <gtest/gtest.h>
#include "engine/Graph.hpp"
#include "engine/GraphIR.hpp"
#include "engine/GraphCompiler.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/ChainFinder.hpp"
#include <algorithm>
#include <random>
#include <set>
#include <sstream>
#include <unordered_set>

using namespace te;

namespace {

// ===========================================================================
// Test fixture library. All types are PurePixel vec4-in / vec4-out
// EXCEPT `boundary_*` which are non-fuseable, and `fanout_*` / `fanin_*`
// which are multi-socket. The `make_*` helpers cover the four PassKind
// barrier classes the algorithm must respect.
// ===========================================================================

NodeType make_type(const std::string& id,
                   uint32_t n_in,
                   uint32_t n_out,
                   uint32_t n_params = 0,
                   PassKind kind    = PassKind::PurePixel) {
    NodeType t;
    t.id = id;
    t.display_name = id;
    t.pass_kind = kind;
    for (uint32_t i = 0; i < n_in; ++i) {
        Socket s; s.name = "in" + std::to_string(i); s.type = SocketType::Vec4;
        t.inputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        Socket s; s.name = "out" + std::to_string(i); s.type = SocketType::Vec4;
        t.outputs.push_back(s);
    }
    for (uint32_t i = 0; i < n_params; ++i) {
        NodeParam p; p.name = "p" + std::to_string(i); t.params.push_back(p);
    }
    return t;
}

NodeLibrary make_lib() {
    NodeLibrary lib;
    lib.add_public(make_type("source",  0, 1));
    lib.add_public(make_type("step",    1, 1));
    lib.add_public(make_type("param",   1, 1, 3));
    lib.add_public(make_type("boundary",1, 1, 0, PassKind::Boundary));
    lib.add_public(make_type("reduce",  1, 1, 0, PassKind::Reduction));
    lib.add_public(make_type("feedback",1, 1, 0, PassKind::Feedback));
    lib.add_public(make_type("readback",1, 1, 0, PassKind::Readback));
    lib.add_public(make_type("debug",   1, 1, 0, PassKind::DebugPreview));
    lib.add_public(make_type("fanout",  1, 4, 0, PassKind::Boundary));
    lib.add_public(make_type("fanin",   4, 1, 0, PassKind::Boundary));
    lib.add_public(make_type("split",   1, 4, 0, PassKind::PurePixel));   // pure multi-output
    return lib;
}

// Holds BOTH the validate_graph IR and the compile result, so tests
// that need to inspect the IR for invariant checks have it locally.
struct CompileFixture {
    bool                ok      = false;
    std::string         error;
    GraphIR             ir;
    CompileGraphResult  compiled;
};

CompileFixture compile_fixture(const Graph& g, const NodeLibrary& lib) {
    CompileFixture f;
    auto vr = validate_graph(g, lib);
    if (!vr.success) {
        f.error = vr.error;
        ADD_FAILURE() << "validate_graph failed: " << vr.error;
        return f;
    }
    f.ir   = vr.ir;
    f.compiled = GraphCompiler::compile(vr.ir, lib);
    f.ok   = f.compiled.success;
    if (!f.ok) {
        f.error = f.compiled.error;
        ADD_FAILURE() << "compile failed: " << f.compiled.error;
    }
    return f;
}

const ValidatedNode* find_in_ir(const GraphIR& ir, NodeId id) { return ir.find(id); }

// ===========================================================================
// INVARIANT ASSERTIONS
//
// These check what MUST be true of any chain decomposition.  They are
// the senior-expert's defence against bad test expectations: if a
// future refactor of find_chains breaks a property, the test fails
// with a clear message, no matter what specific chain count comes out.
// ===========================================================================

// INVARIANT 1: vertex-disjoint cover. Every node in the active subgraph
// appears in EXACTLY ONE chain.
::testing::AssertionResult check_vertex_disjoint_cover(
        const std::vector<Chain>& chains, const GraphIR& ir) {
    std::unordered_set<NodeId> seen;
    for (size_t i = 0; i < chains.size(); ++i) {
        for (NodeId n : chains[i].nodes) {
            if (seen.count(n)) {
                std::ostringstream os;
                os << "node " << n << " appears in chain " << i
                   << " and at least one earlier chain";
                return ::testing::AssertionFailure() << os.str();
            }
            seen.insert(n);
        }
    }
    if (seen.size() != ir.nodes.size()) {
        std::ostringstream os;
        os << "chain cover has " << seen.size() << " nodes but IR has "
           << ir.nodes.size();
        return ::testing::AssertionFailure() << os.str();
    }
    return ::testing::AssertionSuccess();
}

// INVARIANT 2: every chain is non-empty.
::testing::AssertionResult check_chains_nonempty(const std::vector<Chain>& chains) {
    for (size_t i = 0; i < chains.size(); ++i) {
        if (chains[i].nodes.empty()) {
            return ::testing::AssertionFailure() << "chain " << i << " is empty";
        }
    }
    return ::testing::AssertionSuccess();
}

// INVARIANT 3: a chain is either a PurePixel run OR a singleton barrier.
// Equivalently: any chain with size > 1 must consist entirely of nodes
// that are NOT barriers (PurePixel, single-output, in_deg=1 except first,
// out_deg=1 except last).
::testing::AssertionResult check_chain_linearity(
        const std::vector<Chain>& chains, const GraphIR& ir,
        const NodeLibrary& lib) {
    // Reuse the same edge index the algorithm uses so we measure the
    // same thing it does.
    std::unordered_map<NodeId, std::vector<NodeId>> succs;
    std::unordered_map<NodeId, uint32_t> in_deg;
    for (const auto& c : ir.connections) {
        succs[c.src_node].push_back(c.dst_node);
        if (c.dst_socket == 0) ++in_deg[c.dst_node];
    }
    auto is_barrier_fn = [&](NodeId id) -> bool {
        const auto* inst = find_in_ir(ir, id);
        if (!inst) return true;
        if (inst->pass_kind != PassKind::PurePixel) return true;
        const auto* type = lib.find(inst->type_id);
        if (!type || type->outputs.size() != 1) return true;
        auto it = succs.find(id);
        if (it != succs.end() && it->second.size() > 1) return true;
        auto id_it = in_deg.find(id);
        if (id_it != in_deg.end() && id_it->second > 1) return true;
        return false;
    };
    for (size_t i = 0; i < chains.size(); ++i) {
        const auto& ch = chains[i].nodes;
        if (ch.size() == 1) continue;        // singleton: barrier is fine
        for (size_t j = 0; j < ch.size(); ++j) {
            if (is_barrier_fn(ch[j])) {
                std::ostringstream os;
                os << "chain " << i << " has size " << ch.size()
                   << " but contains barrier node " << ch[j] << " at position " << j;
                return ::testing::AssertionFailure() << os.str();
            }
            if (j + 1 < ch.size()) {
                // Interior: must have exactly 1 successor AND that
                // successor must be the next node in the chain.
                if (succs[ch[j]].size() != 1 || succs[ch[j]][0] != ch[j + 1]) {
                    std::ostringstream os;
                    os << "chain " << i << " node " << ch[j]
                       << " does not have exactly one successor pointing to "
                       << ch[j + 1];
                    return ::testing::AssertionFailure() << os.str();
                }
            }
        }
    }
    return ::testing::AssertionSuccess();
}

// INVARIANT 4: chain length respects max_chain_length.
::testing::AssertionResult check_max_length(const std::vector<Chain>& chains,
                                            uint32_t max_len) {
    for (size_t i = 0; i < chains.size(); ++i) {
        if (chains[i].nodes.size() > max_len) {
            std::ostringstream os;
            os << "chain " << i << " has size " << chains[i].nodes.size()
               << " > max_chain_length " << max_len;
            return ::testing::AssertionFailure() << os.str();
        }
    }
    return ::testing::AssertionSuccess();
}

// INVARIANT 5: bypassed flag is set iff any node in the chain is bypassed.
::testing::AssertionResult check_bypassed_propagation(
        const std::vector<Chain>& chains, const GraphIR& ir) {
    for (size_t i = 0; i < chains.size(); ++i) {
        bool any = false;
        for (NodeId n : chains[i].nodes) {
            const auto* inst = find_in_ir(ir, n);
            if (inst && inst->bypassed) { any = true; break; }
        }
        if (chains[i].bypassed != any) {
            std::ostringstream os;
            os << "chain " << i << " bypassed=" << chains[i].bypassed
               << " but actual=" << any;
            return ::testing::AssertionFailure() << os.str();
        }
    }
    return ::testing::AssertionSuccess();
}

// INVARIANT 6: param_offsets[i] = sum of params in nodes[0..i-1].
::testing::AssertionResult check_param_offsets(
        const std::vector<Chain>& chains, const GraphIR& ir,
        const NodeLibrary& lib) {
    for (size_t i = 0; i < chains.size(); ++i) {
        const auto& ch = chains[i];
        if (ch.param_offsets.size() != ch.nodes.size()) {
            return ::testing::AssertionFailure()
                << "chain " << i << " param_offsets.size="
                << ch.param_offsets.size() << " != nodes.size="
                << ch.nodes.size();
        }
        uint32_t expected = 0;
        for (size_t j = 0; j < ch.nodes.size(); ++j) {
            if (ch.param_offsets[j] != expected) {
                return ::testing::AssertionFailure()
                    << "chain " << i << " node " << j
                    << " param_offsets=" << ch.param_offsets[j]
                    << " expected=" << expected;
            }
            const auto* inst = find_in_ir(ir, ch.nodes[j]);
            const auto* type = inst ? lib.find(inst->type_id) : nullptr;
            expected += type ? static_cast<uint32_t>(type->params.size()) : 0u;
        }
        if (ch.total_params != expected) {
            return ::testing::AssertionFailure()
                << "chain " << i << " total_params=" << ch.total_params
                << " expected=" << expected;
        }
    }
    return ::testing::AssertionSuccess();
}

// Run the full invariant battery. Returns AssertionResult for one-shot
// use inside EXPECT_TRUE so a failure message names the violation.
::testing::AssertionResult check_all_invariants(
        const std::vector<Chain>& chains, const GraphIR& ir,
        const NodeLibrary& lib, uint32_t max_len) {
    auto r1 = check_vertex_disjoint_cover(chains, ir);  if (!r1) return r1;
    auto r2 = check_chains_nonempty(chains);            if (!r2) return r2;
    auto r3 = check_chain_linearity(chains, ir, lib);   if (!r3) return r3;
    auto r4 = check_max_length(chains, max_len);        if (!r4) return r4;
    auto r5 = check_bypassed_propagation(chains, ir);   if (!r5) return r5;
    auto r6 = check_param_offsets(chains, ir, lib);     if (!r6) return r6;
    return ::testing::AssertionSuccess();
}

// ===========================================================================
// REFERENCE ORACLE: a SIMPLE alternative algorithm that should produce
// the SAME number of chains as find_chains for "well-behaved" graphs
// (no fan-in, no fan-out, all PurePixel 1-in-1-out). We use this as a
// cross-check on the algorithm's counting.
//
// Algorithm "every node is a barrier": produces |V| chains, the trivial
// upper bound. The non-trivial lower bound for our problem is
//   |V| - |max_matching|   (Dilworth's theorem for DAGs)
// and our greedy should be close to it for linear-only DAGs.
// ===========================================================================

// Reference: every node is its own chain.
size_t oracle_count_every_singleton(const GraphIR& ir) {
    return ir.nodes.size();
}

// Reference: min path cover LOWER BOUND for the linear-only subgraph.
// Implemented as bipartite matching: left = nodes, right = nodes,
// edge (u, v) iff there is a connection u->v AND u has out_deg=1 AND
// v has in_deg=1 (so this edge can be the "interior" of a chain).
// Max matching = max number of consecutive pairs we can join.
// Result = |V| - max_matching.
size_t oracle_min_chain_count_lower_bound(const GraphIR& ir) {
    const size_t n = ir.nodes.size();
    if (n == 0) return 0;
    std::unordered_map<NodeId, size_t> id_to_idx;
    for (size_t i = 0; i < n; ++i) id_to_idx[ir.nodes[i].id] = i;
    std::unordered_map<NodeId, uint32_t> in_deg, out_deg;
    for (const auto& c : ir.connections) {
        if (c.dst_socket == 0) ++in_deg[c.dst_node];
        ++out_deg[c.src_node];
    }
    // Adjacency: left u -> right v iff u->v edge exists and u is
    // out_deg=1, v is in_deg=1.
    std::vector<std::vector<size_t>> adj(n);
    for (const auto& c : ir.connections) {
        if (c.dst_socket != 0) continue;
        if (out_deg[c.src_node] != 1) continue;
        if (in_deg[c.dst_node] != 1) continue;
        size_t u = id_to_idx.at(c.src_node);
        size_t v = id_to_idx.at(c.dst_node);
        adj[u].push_back(v);
    }
    // Bipartite matching via augmenting paths (Kuhn), O(VE).
    std::vector<size_t> match_r(n, SIZE_MAX);
    auto try_augment = [&](auto& self, size_t u,
                           std::vector<bool>& seen) -> bool {
        for (size_t v : adj[u]) {
            if (seen[v]) continue;
            seen[v] = true;
            if (match_r[v] == SIZE_MAX || self(self, match_r[v], seen)) {
                match_r[v] = u;
                return true;
            }
        }
        return false;
    };
    size_t max_match = 0;
    for (size_t u = 0; u < n; ++u) {
        std::vector<bool> seen(n, false);
        if (try_augment(try_augment, u, seen)) ++max_match;
    }
    return n - max_match;
}

// ===========================================================================
// RANDOM DAG GENERATOR (for fuzz testing).
//
// Generates a DAG with `n` PurePixel 1-in-1-out nodes plus optional
// barriers, then connects them respecting the active-subgraph BFS:
// every node must be reachable from the output. We achieve this by
// always building the graph forward (each new node picks predecessors
// from existing nodes, then we mark one node as output).
// ===========================================================================

struct FuzzGraph {
    Graph g;
    std::vector<NodeId> barrier_ids;
    std::vector<NodeId> source_ids;
};

FuzzGraph make_random_dag(std::mt19937& rng, size_t target_nodes) {
    FuzzGraph out;
    auto& g = out.g;
    NodeId next_id = 1;
    std::vector<NodeId> ids;
    for (size_t i = 0; i < target_nodes; ++i) {
        NodeId id = next_id++;
        ids.push_back(id);
        g.nodes.push_back({id, "step"});
        // Every ~7th node is a barrier so we exercise both paths.
        if (std::uniform_int_distribution<int>(0, 6)(rng) == 0) {
            g.nodes.back().type_id = "boundary";
            out.barrier_ids.push_back(id);
        } else {
            out.source_ids.push_back(id);
        }
    }
    if (ids.empty()) return out;       // edge case
    // Each new node (after the first) connects to 1-2 earlier nodes
    // (uniformly). This produces a realistic graph with both linear
    // segments and forks. The last node is the output.
    for (size_t i = 1; i < ids.size(); ++i) {
        NodeId dst = ids[i];
        size_t n_pred = std::uniform_int_distribution<size_t>(1, 2)(rng);
        std::set<size_t> picked;
        while (picked.size() < std::min(n_pred, i)) {
            picked.insert(std::uniform_int_distribution<size_t>(0, i - 1)(rng));
        }
        for (size_t p : picked) {
            g.connections.push_back({ids[p], 0, dst, 0});
        }
    }
    // Mark the last node as the output. To ensure the whole graph is
    // reachable from the output (i.e. validator's active-subgraph BFS
    // keeps all nodes), verify that the first node reaches the last
    // transitively; if not, add a bridge edge.
    g.output_node = ids.back();
    std::unordered_set<NodeId> descendants{ids.back()};
    std::vector<NodeId> stack{ids.back()};
    while (!stack.empty()) {
        NodeId cur = stack.back(); stack.pop_back();
        for (const auto& c : g.connections) {
            if (c.dst_node == cur && !descendants.count(c.src_node)) {
                descendants.insert(c.src_node);
                stack.push_back(c.src_node);
            }
        }
    }
    if (!descendants.count(ids.front())) {
        // Force connectivity: add an edge from first to last.
        g.connections.push_back({ids.front(), 0, ids.back(), 0});
    }
    return out;
}

} // namespace

// ===========================================================================
// HAPPY-PATH EXAMPLE TESTS (specific assertions, not just invariants)
// ===========================================================================

TEST(ChainFinding, LinearChainFuses) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    // 3 PurePixel nodes in a line, no barriers: exactly 1 chain.
    EXPECT_EQ(f.compiled.pass_plan.chains.size(), 1u);
    EXPECT_EQ(f.compiled.pass_plan.chains[0].nodes.size(), 3u);
}

TEST(ChainFinding, BoundaryBreaksChain) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "boundary"});
    g.nodes.push_back({3, "step"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    // source->boundary->step: 3 chains. The boundary becomes a
    // singleton; source and step each get their own chains because
    // boundary is a barrier.
    EXPECT_EQ(f.compiled.pass_plan.chains.size(), 3u);
}

TEST(ChainFinding, MultiOutputPurePixelIsBarrier) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "split"});      // PurePixel with 4 outputs
    g.nodes.push_back({3, "step"});
    g.nodes.push_back({4, "step"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({2, 1, 4, 0});
    // Fan-in the two step branches so both are reachable from output.
    g.nodes.push_back({5, "fanin"});      // 4-in boundary
    g.connections.push_back({3, 0, 5, 0});
    g.connections.push_back({4, 0, 5, 1});
    g.output_node = 5;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    // [source, 1] + [2 split singleton] + [3, 1] + [4, 1] + [5 fanin singleton] = 5
    // In our linear-counting test we just need invariant correctness.
    EXPECT_TRUE(check_all_invariants(f.compiled.pass_plan.chains, f.ir,
                                     lib, 64));
}

// ===========================================================================
// INVARIANT-ONLY TESTS: shape the algorithm by what MUST be true.
// ===========================================================================

TEST(ChainFinding, DiamondTopoIsCovered) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    g.nodes.push_back({4, "fanin"});  // barrier
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 4, 0});
    g.connections.push_back({3, 0, 4, 1});
    g.output_node = 4;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    // All 4 nodes must appear in chains. source and fanin are barriers
    // (fan-out and fan-in respectively) so they get singleton chains.
    // Steps 2 and 3 have barriers on both sides -> singleton each.
    // Total = 4 chains. The exact count is an emergent property of
    // the algorithm; we only require invariants here.
    EXPECT_TRUE(check_all_invariants(f.compiled.pass_plan.chains, f.ir,
                                     lib, 64));
}

TEST(ChainFinding, FanInBarrierAloneIsSingleton) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "fanin"});  // 2-in barrier = output
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 3, 1});
    g.output_node = 3;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    EXPECT_TRUE(check_all_invariants(f.compiled.pass_plan.chains, f.ir,
                                     lib, 64));
    // Stronger: 1 (source) and 2 (step) are independent predecessors
    // of fanin. 1 is a fan-out (in_deg=0, out_deg=2): singleton.
    // 2 has 1 pred (1=barrier) and 1 succ (3=barrier): singleton.
    // 3 is fan-in: singleton. Total = 3.
    EXPECT_EQ(f.compiled.pass_plan.chains.size(), 3u);
}

TEST(ChainFinding, FanOutBarrierAloneIsSingleton) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({1, 0, 3, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.output_node = 3;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    EXPECT_TRUE(check_all_invariants(f.compiled.pass_plan.chains, f.ir,
                                     lib, 64));
    // 1 fan-out barrier, 2 step, 3 fan-in barrier. 3 chains.
    EXPECT_EQ(f.compiled.pass_plan.chains.size(), 3u);
}

TEST(ChainFinding, LongChainRespectsMaxLength) {
    auto lib = make_lib();
    Graph g;
    constexpr int N = 100;
    for (int i = 1; i <= N; ++i) g.nodes.push_back({(uint32_t)i, "step"});
    for (int i = 1; i < N; ++i) g.connections.push_back({(uint32_t)i, 0, (uint32_t)(i + 1), 0});
    g.output_node = N;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    EXPECT_TRUE(check_all_invariants(f.compiled.pass_plan.chains, f.ir,
                                     lib, 64));
    size_t total = 0;
    for (const auto& ch : f.compiled.pass_plan.chains) total += ch.nodes.size();
    EXPECT_EQ(total, (size_t)N);
}

TEST(ChainFinding, MaxChainLengthOptionRespected) {
    auto lib = make_lib();
    Graph g;
    for (int i = 1; i <= 10; ++i) g.nodes.push_back({(uint32_t)i, "step"});
    for (int i = 1; i < 10; ++i) g.connections.push_back({(uint32_t)i, 0, (uint32_t)(i + 1), 0});
    g.output_node = 10;
    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success);
    ChainFinderOptions opts; opts.max_chain_length = 2;
    auto chains = find_chains(r.ir, lib, opts);
    EXPECT_TRUE(check_max_length(chains, 2u));
    size_t total = 0; for (const auto& ch : chains) total += ch.nodes.size();
    EXPECT_EQ(total, 10u);
}

TEST(ChainFinding, ParamOffsetsAreCorrect) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "param"});     // 3 params
    g.nodes.push_back({3, "step"});      // 0 params
    g.nodes.push_back({4, "param"});     // 3 params
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.output_node = 4;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    ASSERT_EQ(f.compiled.pass_plan.chains.size(), 1u);
    EXPECT_EQ(f.compiled.pass_plan.chains[0].param_offsets, (std::vector<uint32_t>{0, 0, 3, 3}));
    EXPECT_EQ(f.compiled.pass_plan.chains[0].total_params, 6u);
}

TEST(ChainFinding, BypassedNodeMakesChainBypassed) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.nodes.back().bypassed = true;
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    ASSERT_EQ(f.compiled.pass_plan.chains.size(), 1u);
    EXPECT_TRUE(f.compiled.pass_plan.chains[0].bypassed);
}

TEST(ChainFinding, PassKindClassesAllBarrier) {
    // For each non-PurePixel PassKind, a 3-node graph with that node
    // in the middle should produce 3 chains.
    std::vector<PassKind> barrier_kinds = {
        PassKind::Boundary, PassKind::Reduction, PassKind::Feedback,
        PassKind::Readback,  PassKind::DebugPreview
    };
    for (PassKind k : barrier_kinds) {
        auto lib = make_lib();
        NodeType t = make_type("intruder", 1, 1, 0, k);
        lib.add_public(t);
        Graph g;
        g.nodes.push_back({1, "source"});
        g.nodes.push_back({2, "intruder"});
        g.nodes.push_back({3, "step"});
        g.connections.push_back({1, 0, 2, 0});
        g.connections.push_back({2, 0, 3, 0});
        g.output_node = 3;
        auto f = compile_fixture(g, lib);
        ASSERT_TRUE(f.ok) << "compile failed for PassKind="
                          << static_cast<int>(k);
        EXPECT_EQ(f.compiled.pass_plan.chains.size(), 3u)
            << "PassKind " << static_cast<int>(k) << " should be a barrier";
    }
}

TEST(ChainFinding, EmptyGraphRejectedUpstream) {
    auto lib = make_lib();
    Graph g;
    auto r = validate_graph(g, lib);
    EXPECT_FALSE(r.success);
    // Defensive: if validate allowed it, find_chains returns empty.
    if (r.success) {
        EXPECT_EQ(find_chains(r.ir, lib).size(), 0u);
    }
}

TEST(ChainFinding, OutputNodePurePixelEndsChain) {
    auto lib = make_lib();
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "step"});
    g.connections.push_back({1, 0, 2, 0});
    g.output_node = 2;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    // The output_node is NOT a barrier purely by virtue of being the
    // output. Step 2 is PurePixel, in_deg=1, out_deg=0, 1 output, so
    // it is part of the chain. Total = 1 chain.
    EXPECT_EQ(f.compiled.pass_plan.chains.size(), 1u);
    EXPECT_EQ(f.compiled.pass_plan.chains[0].nodes.size(), 2u);
}

// ===========================================================================
// CROSS-VALIDATION: for the "linear-only" subgraph (all PurePixel
// 1-in-1-out, no multi-output, no fan-in, no fan-out) the algorithm
// should produce the MINIMUM number of chains, equal to |V| -
// max_matching (Dilworth's theorem). We compare against this lower
// bound.
// ===========================================================================

TEST(ChainFinding, LinearOnlyMatchesMinPathCover) {
    auto lib = make_lib();
    // A linear-only DAG: 1->2->3, 1->4, 4->5, 5->3.  Three linear
    // paths from 1: [1,2,3], [1,4,5,3].  But 3 has in_deg=2 so it's a
    // barrier in the FULL graph.  Pick a DAG that is *truly* linear:
    // 1->2->3->4->5 with one extra branch 2->5 (skip 3,4).
    Graph g;
    g.nodes.push_back({1, "step"});
    g.nodes.push_back({2, "step"});
    g.nodes.push_back({3, "step"});
    g.nodes.push_back({4, "step"});
    g.nodes.push_back({5, "step"});
    g.connections.push_back({1, 0, 2, 0});
    g.connections.push_back({2, 0, 3, 0});
    g.connections.push_back({3, 0, 4, 0});
    g.connections.push_back({4, 0, 5, 0});
    g.output_node = 5;
    auto f = compile_fixture(g, lib);
    ASSERT_TRUE(f.ok);
    EXPECT_TRUE(check_all_invariants(f.compiled.pass_plan.chains, f.ir,
                                     lib, 64));
    // In a true linear chain the algorithm should produce 1 chain.
    EXPECT_EQ(f.compiled.pass_plan.chains.size(), 1u);
    size_t lb = oracle_min_chain_count_lower_bound(f.ir);
    EXPECT_GE(f.compiled.pass_plan.chains.size(), lb);
}

// ===========================================================================
// PROPERTY-BASED FUZZ: generate random DAGs, check invariants, also
// cross-validate against the min-path-cover lower bound.
// ===========================================================================

TEST(ChainFinding, FuzzRandomDagsRespectAllInvariants) {
    auto lib = make_lib();
    std::mt19937 rng(0xC0FFEE);
    int total_graphs = 0;
    for (int trial = 0; trial < 200; ++trial) {
        size_t n = std::uniform_int_distribution<size_t>(1, 30)(rng);
        FuzzGraph fg = make_random_dag(rng, n);
        if (fg.g.nodes.empty()) continue;
        auto r = validate_graph(fg.g, lib);
        if (!r.success) continue;       // cycle, etc. -- skip, not a bug
        auto chains = find_chains(r.ir, lib);
        EXPECT_TRUE(check_all_invariants(chains, r.ir, lib, 64))
            << "Fuzz trial " << trial << " (n=" << n << ") failed invariants";
        EXPECT_GE(chains.size(), oracle_min_chain_count_lower_bound(r.ir))
            << "Fuzz trial " << trial << " produced fewer chains than the lower bound";
        ++total_graphs;
    }
    EXPECT_GE(total_graphs, 100) << "Fuzz skipped too many graphs";
}

TEST(ChainFinding, FuzzRandomDagsRespectCap) {
    auto lib = make_lib();
    std::mt19937 rng(0xBADF00D);
    for (int trial = 0; trial < 100; ++trial) {
        size_t n = std::uniform_int_distribution<size_t>(20, 100)(rng);
        FuzzGraph fg = make_random_dag(rng, n);
        if (fg.g.nodes.empty()) continue;
        auto r = validate_graph(fg.g, lib);
        if (!r.success) continue;
        ChainFinderOptions opts;
        opts.max_chain_length = std::uniform_int_distribution<uint32_t>(1, 8)(rng);
        auto chains = find_chains(r.ir, lib, opts);
        EXPECT_TRUE(check_max_length(chains, opts.max_chain_length))
            << "Trial " << trial << " cap=" << opts.max_chain_length;
        EXPECT_TRUE(check_vertex_disjoint_cover(chains, r.ir))
            << "Trial " << trial << " cap=" << opts.max_chain_length;
    }
}

TEST(ChainFinding, FuzzEveryNodeCoveredUnderCap) {
    // For a wide variety of caps, the total node count across chains
    // must always equal the active-subgraph node count.
    auto lib = make_lib();
    std::mt19937 rng(42);
    for (int trial = 0; trial < 50; ++trial) {
        size_t n = std::uniform_int_distribution<size_t>(5, 50)(rng);
        FuzzGraph fg = make_random_dag(rng, n);
        if (fg.g.nodes.empty()) continue;
        auto r = validate_graph(fg.g, lib);
        if (!r.success) continue;
        for (uint32_t cap : {1u, 2u, 3u, 7u, 16u, 64u, 1024u}) {
            ChainFinderOptions opts; opts.max_chain_length = cap;
            auto chains = find_chains(r.ir, lib, opts);
            size_t total = 0;
            for (const auto& ch : chains) total += ch.nodes.size();
            EXPECT_EQ(total, r.ir.nodes.size())
                << "Trial " << trial << " cap=" << cap;
        }
    }
}

// ===========================================================================
// STAGE 4.2: Multi-input node with one in-graph predecessor
//
// A PurePixel node with N input sockets where:
//   - exactly ONE socket is connected to a chain predecessor
//   - the other N-1 sockets are unconnected
// has in_degree=1 in the graph (the chain finder counts only
// dst_socket==0 edges; for a multi-input node with 1 connection, the
// 1 connection is at socket 0, so in_degree=1).
//
// Such a node is NOT a barrier (in_degree=1 is below the >1 fan-in
// threshold) and is chainable. Stage 4.2 lifts the emitter's previous
// "multi-input rejected" rule to make this case work end-to-end.
// ===========================================================================

TEST(ChainFinding, MultiInputNodeJoinsChain) {
    // source(1) -> blend(2). blend has 2 inputs; .a = source, .b
    // is unconnected. Expected: ONE chain [source, blend] (not two
    // singletons, and not a barrier for blend).
    auto lib = make_lib();
    lib.add_public(make_type("blend", 2, 1, 0, PassKind::PurePixel));
    Graph g;
    g.nodes.push_back({1, "source"});
    g.nodes.push_back({2, "blend"});
    g.connections.push_back({1, 0, 2, 0});   // source -> blend.a
    // blend.b is unconnected
    g.output_node = 2;
    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    auto chains = find_chains(r.ir, lib);
    ASSERT_EQ(chains.size(), 1u)
        << "Multi-input node with one in-graph predecessor should join the chain";
    EXPECT_EQ(chains[0].nodes, (std::vector<NodeId>{1, 2}));
    // And the vertex-disjoint cover invariant still holds.
    EXPECT_TRUE(check_vertex_disjoint_cover(chains, r.ir));
}

TEST(ChainFinding, MultiInputHeadIsSingleton) {
    // blend(2) is the head: both inputs are unconnected, no in-graph
    // predecessor. Expected: ONE chain [blend] (singleton). The chain
    // shader will fetch both inputs from pc.in_sampled_slots[].
    auto lib = make_lib();
    lib.add_public(make_type("blend", 2, 1, 0, PassKind::PurePixel));
    Graph g;
    g.nodes.push_back({1, "blend"});
    g.output_node = 1;
    auto r = validate_graph(g, lib);
    ASSERT_TRUE(r.success) << r.error;
    auto chains = find_chains(r.ir, lib);
    ASSERT_EQ(chains.size(), 1u);
    EXPECT_EQ(chains[0].nodes, (std::vector<NodeId>{1}));
}
