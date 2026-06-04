#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PassPlan.hpp"
#include <cstdint>
#include <vector>

namespace te {

// ---------------------------------------------------------------------------
// ChainFinder -- Stage 3 of the pass-fusion plan.
//
// Walks the validated graph and groups adjacent PurePixel nodes into
// maximal linear chains. Non-PurePixel nodes (Boundary, Reduction, etc.),
// multi-output nodes, fan-in, fan-out, and the output_node are all chain
// boundaries. Each chain is a vertex-disjoint subset of the graph; every
// node appears in exactly one chain.
//
// The algorithm is pure graph topology -- no VRAM, no VMA, no resource
// manager. The chain plan IS the resource plan: each chain produces one
// output image; pure-internal PurePixel nodes produce no image. See
// 06_chain_finding_research.md for the algorithm choice rationale.
//
// Time complexity: O(V + E).
// Space complexity: O(V).
// ---------------------------------------------------------------------------

struct ChainFinderOptions {
    // Maximum nodes in a single fused chain. Caps register pressure
    // (see 06_chain_finding_research.md section 5). 64 is conservative;
    // matches Material Maker's ~50-100 and TVM's cost-based defaults.
    uint32_t max_chain_length = 64;

    // When true, log a warning for any node in a cycle (Phase 1 treats
    // cycles as barriers and falls back to singletons).
    bool log_cycles = true;
};

// Returns the chains. The vector order is stable: chains are produced
// in topological order of their first node.
std::vector<Chain> find_chains(const GraphIR& ir,
                               const NodeLibrary& lib,
                               const ChainFinderOptions& opts = {});

} // namespace te
