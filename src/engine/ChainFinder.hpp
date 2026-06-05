#pragma once
#include "engine/GraphIR.hpp"
#include "engine/NodeLibrary.hpp"
#include "engine/PassPlan.hpp"
#include <cstdint>
#include <vector>

namespace te {

// ChainFinder -- Stage 3 of pass-fusion plan. Walks the validated graph and groups adjacent PurePixel nodes into maximal linear chains. Non-PurePixel nodes, multi-output nodes, fan-in/out, and output_node are chain boundaries. Each chain is vertex-disjoint; every node appears in exactly one chain. O(V+E) time, O(V) space.

struct ChainFinderOptions {
    // Maximum nodes in a single fused chain. Caps register pressure. 64 is conservative.
    uint32_t max_chain_length = 64;

    // When true, log warning for nodes in cycles (cycles become barriers, falling back to singletons).
    bool log_cycles = true;
};

// Returns the chains in stable topological order of their first node.
std::vector<Chain> find_chains(const GraphIR& ir,
                               const NodeLibrary& lib,
                               const ChainFinderOptions& opts = {});

} // namespace te
