# Implementation Plan - GPU Register Allocation & Graph Coloring for Node Graph Fusion

## 1. Workflow / Pipeline (3D Artist / Developer View)
In TextureSynth, a procedural texture is defined as a node **graph**. Normally, every **node** executes as a separate GPU **dispatch** pass, writing to and reading from intermediate **images**. To optimize this, the **chain** fusion system collapses eligible node chains into a single compute shader. This runs in one **dispatch**, keeping intermediates in GPU registers rather than writing to texture memory. However, register space is limited. If a chain uses too many registers, we must split it. We use a **slider** in the Blender **node editor** to control the register budget. 

This optimization uses **graph coloring** to reuse registers. Below is a comparison of the pipeline before and after this optimization:

| Stage | Before (Additive Allocation) | After (Graph Coloring Allocation) |
|---|---|---|
| **Liveness Analysis** | None (All variables assumed live forever) | Computes precise start/end pass for each node output |
| **Interference** | None (Costs are summed: $A + B + C + D$) | Builds conflict graph of variables live at the same time |
| **Coloring / Reuse** | Every output gets a unique local variable | Solves coloring so non-overlapping variables share registers |
| **Fusion Capacity** | Low (Chains split early due to false pressure) | High (Chains can be much larger, minimizing dispatches) |

---

## 2. Architecture / Data Flow
We will introduce a new directory `src/engine/regalloc/` to house the register allocator. It will be completely isolated from the current execution pipeline to prevent any breakage, but it will be designed to directly integrate with the existing `te::fusion` and `te::reg` namespaces later.

The core components will map to these new files:
- [Liveness.hpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/regalloc/Liveness.hpp) & [Liveness.cpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/regalloc/Liveness.cpp): Computes live ranges for all node output variables in a topological sequence.
- [InterferenceGraph.hpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/regalloc/InterferenceGraph.hpp) & [InterferenceGraph.cpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/regalloc/InterferenceGraph.cpp): Builds the conflict graph where nodes represent variables and edges represent overlapping lifetimes.
- [Coloring.hpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/regalloc/Coloring.hpp) & [Coloring.cpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/regalloc/Coloring.cpp): Implements Chaitin's graph-coloring heuristic (optimistic coloring) and Linear Scan (greedy interval coloring) to assign virtual registers to each variable.

### Integration Connections:
- `te::fusion::FusionPlanner` (in [FusionPlanner.hpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/graphfusion/FusionPlanner.hpp#L31)): In the future, this planner can query `regalloc` to check if a subset of nodes fits in the register budget, allowing larger chains to fuse.
- `te::glsl::GlslBuilder` (in [GlslBuilder.hpp](file:///c:/Users/User/Documents/0/TEXTURESYNTH/src/engine/graphfusion/GlslBuilder.hpp#L12)): Instead of declaring `_local_0`, `_local_1`, etc., the generator can map each local variable to `r[color]`.

---

## 3. Code/Pseudo-code Example

### 3.1 Live Interval Computation (C++)
```cpp
struct LiveInterval {
    uint32_t start = 0; // Topological step where variable is defined
    uint32_t end = 0;   // Last topological step where variable is read
};

// Given a topological sequence of nodes and their connections:
std::unordered_map<NodeId, LiveInterval> compute_intervals(
    const std::vector<NodeId>& nodes,
    const ConnByDst& connections,
    NodeId output_node) 
{
    std::unordered_map<NodeId, LiveInterval> intervals;
    std::unordered_map<NodeId, uint32_t> node_to_step;
    for (uint32_t step = 0; step < nodes.size(); ++step) {
        node_to_step[nodes[step]] = step;
        intervals[nodes[step]].start = step;
        intervals[nodes[step]].end = step; // default to self-contained
    }
    
    // Scan connections to extend end steps
    for (uint32_t step = 0; step < nodes.size(); ++step) {
        NodeId dst = nodes[step];
        if (connections.count(dst)) {
            for (const auto& [dst_socket, src] : connections.at(dst)) {
                if (node_to_step.count(src)) {
                    intervals[src].end = std::max(intervals[src].end, step);
                }
            }
        }
    }
    // Output node is live until the end of execution
    intervals[output_node].end = static_cast<uint32_t>(nodes.size());
    return intervals;
}
```

### 3.2 Resulting Fused GLSL Structure (GLSL)
Instead of:
```glsl
void main() {
    vec4 _local_0; // output of node 0
    vec4 _local_1; // output of node 1
    vec4 _local_2; // output of node 2
    // ...
}
```
We generate:
```glsl
void main() {
    // Only two registers needed because local_0 and local_2 lifetimes don't overlap!
    vec4 r0; 
    vec4 r1;

    r0 = node_source(uv);                   // _local_0 mapped to r0
    r1 = node_step(uv, r0, param1);         // _local_1 mapped to r1 (r0 is read)
    r0 = node_step(uv, r1, param2);         // _local_2 mapped to r0 (reused!)
    
    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, r0);
}
```

---

## 4. Verification Plan

### Automated Tests
We will write a standalone unit test binary/source file `src/engine/regalloc/TestRegAlloc.cpp` containing a main function that:
- Defines various node graph topologies (linear, fan-out/fan-in, multiple branches).
- Performs liveness analysis and computes live intervals.
- Colors the resulting interference graphs.
- Verifies that overlapping variables never get assigned the same register (color), and that the total number of colors matches the theoretical minimum.

We will add a build script or compile instructions to compile this test.
