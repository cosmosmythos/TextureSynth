# register_allocation — GPU Register Allocation & Graph Coloring for Node Graph Fusion

## 1. Purpose
Graph-coloring register allocator for fused compute shader chains. Assigns live node outputs to a limited pool of `vec4` GLSL locals, minimizing register pressure so longer chains can fit in a single dispatch without spilling to shared memory.

## 2. Ownership
All files under `src/engine/register_allocation/`. Namespace: `te::register_allocation`.

| File | Role |
|---|---|
| `LivenessAnalysis.hpp/cpp` | Computes live intervals (start/end topological step) for each `ResourceUUID`. |
| `InterferenceGraph.hpp/cpp` | Builds undirected conflict graph from overlapping intervals. Nodes = resources, edges = interference. |
| `GraphColorer.hpp/cpp` | Two coloring strategies: Chaitin-Briggs (optimistic) and Linear Scan (greedy interval). |

## 3. Workflow / Pipeline (3D Artist / Developer View)
In TextureSynth, a procedural texture is a node **graph**. Normally each **node** executes as a separate GPU **dispatch**, writing to intermediate **images**. **Chain** fusion collapses eligible chains into a single compute shader running in one **dispatch**, keeping intermediates in GPU registers. Register space is limited; if a chain exceeds the budget, it must be split. A **slider** in the Blender **node editor** controls the register budget.

Graph coloring reuses registers across non-overlapping lifetimes:

| Stage | Before (Additive Allocation) | After (Graph Coloring Allocation) |
|---|---|---|
| **Liveness Analysis** | None (all variables live forever) | Computes precise start/end pass per node output |
| **Interference** | None (costs summed: A + B + C + D) | Builds conflict graph of variables live simultaneously |
| **Coloring / Reuse** | Every output gets unique local | Non-overlapping variables share registers |
| **Fusion Capacity** | Low (chains split early) | High (much larger chains, fewer dispatches) |

---

## 4. Architecture / Data Flow

### Integration Connections:
- `te::fusion::FusionPlanner` (in [FusionPlanner.hpp](../graphfusion/FusionPlanner.hpp)): queries `register_allocation` to check if a chain fits the register budget before fusing.
- `te::glsl::GlslBuilder` (in [GlslBuilder.hpp](../graphfusion/GlslBuilder.hpp)): maps each local variable to `r[color]` instead of `_local_0`, `_local_1`, etc.

### Pipeline:
```
Topological order + connections
        |
        v
LivenessAnalysis::compute_intervals()  -->  IntervalMap
        |
        v
InterferenceGraph::build_from_intervals()  -->  InterferenceGraph
        |
        +--[Chaitin-Briggs]--> GraphColorer::color_chaitin_briggs()
        |                          (general, O(N^2) simplify+select)
        |
        +--[Linear Scan]-----> GraphColorer::color_linear_scan()
                                   (optimal for interval graphs, O(N log N))
        |
        v
ColoringResult { assignment, colors_used, spilled }
```

---

## 5. Algorithm Details

### 5.1 Liveness Analysis (LivenessAnalysis.cpp)
Three-pass backward scan over topological order:
1. Create intervals at each source definition site.
2. Extend intervals at each consumer (read) site to max consumer step.
3. Pin external outputs (chain tail, final output) to end-of-sequence.

Complexity: O(N + E) where N = nodes, E = connections.

### 5.2 Interference Graph (InterferenceGraph.cpp)
Sweep-line O(N^2) overlap test. For chains < 50 nodes this is fine. The graph is undirected: `add_edge(a,b)` and `add_edge(b,a)`.

### 5.3 Chaitin-Briggs — Optimistic Graph Coloring (GraphColorer.cpp:34-104)

**Simplify phase**: while graph non-empty:
1. If any node has degree < K (budget), remove it and push onto stack. (Guaranteed colorable: at most K-1 neighbors already colored when popped.)
2. Else — Briggs' **optimistic** improvement: pick the highest-degree node and push anyway. (When popped during Select, its neighbors may have received fewer than K distinct colors.)

**Select phase**: pop from stack, assign lowest color not used by already-colored neighbors. If no color < K available, mark as spilled.

**Why optimistic matters for GPU shaders**: Spilling (writing vec4 to shared memory) is extremely expensive. Classic Chaitin eagerly spills any node with degree >= K. Briggs defers the decision to Select time, discovering that many "high-degree" nodes actually color successfully because neighbors reuse colors.

The simplify-select-spill loop:
```
Simplify:  push low-degree nodes first (safe), then optimistic high-degree (may spill)
Select:    pop in reverse, assign colors greedily
Spill:     only if truly uncolorable during Select (not during Simplify)
```

### 5.4 Linear Scan — Greedy Interval Coloring (GraphColorer.cpp:126-215)

From Poletto & Sarkar (1999), adapted for DAG topological order:

1. Sort intervals by start_step (ties by end_step).
2. Maintain an "active" set of intervals occupying a register, sorted by end_step.
3. For each interval:
   a. **Expire**: remove all active intervals whose end < current start. Return their registers to free pool.
   b. If free pool non-empty: assign lowest free register.
   c. Else if budget available: allocate new register.
   d. Else: **spill** the active interval with the latest end (frees register for shorter-lived current interval).

**Lifetime holes**: Linear scan handles them via the Expire step — when an interval ends, its register returns to the free pool. No explicit "hole" tracking needed. The second-chance variants (not implemented here) would re-assign freed registers to new intervals without requiring a new register.

### 5.5 Register Coalescing (Future Work)
George & Appel's iterated register coalescing: two variables can share a register if they don't interfere AND coalescing doesn't increase the graph's chromatic number (i.e., their neighbors don't form a clique). In TextureSynth, this applies when node A's output feeds only node B and neither interferes with B's other inputs — A and B's outputs could potentially share a register.

---

## 6. GPU-Specific Concerns

### Register Pressure vs Occupancy
- AMD GCN/RDNA: 64 VGPRs per SIMD, 32 SGPRs per wavefront. Occupancy = f(VGPRs used). Fewer VGPRs = more concurrent wavefronts = better latency hiding.
- NVIDIA (SM 7.x+): 255 VGPRs per SM partitioned into 32-slot register files. Occupancy limited by `registers_per_thread * threads_per_block`.
- SPIR-V compilers (Mesa RADV, NVIDIA proprietary) handle final physical register allocation. Our allocator targets **virtual register** count (GLSL locals), which directly maps to VGPR consumption.

### VGPR vs SGPR Distinction
- **VGPR** (Vector General Purpose Register): holds per-thread data (vec4 locals, computed values). Our allocator maps to these.
- **SGPR** (Scalar General Purpose Register): holds uniform data (constants, loop bounds, buffer addresses). Not under our control — SPIR-V compiler allocates these from push constants and uniform expressions.

### Spilling to Shared Memory (LDS)
When a fused shader exceeds the register budget:
1. The allocator returns `spilled` resources in `ColoringResult`.
2. `GlslBuilder` generates store/load to shared memory (`shared vec4 spilled_X`) for spilled variables.
3. LDS bandwidth is ~4x worse than register file but still 100x better than texture memory round-trips.
4. AMD: LDS is 64KB per CU, shared among wavefronts. NVIDIA: shared memory is per-SM, configurable via launch config.

### Fallback Strategy
If spilling count exceeds a threshold, `FusionPlanner` splits the chain back into multiple dispatch passes — the additive-allocation fallback.

---

## 7. Perfect Graphs & Interval Graphs — Theoretical Foundation

### Why This Matters
The interference graph produced by TextureSynth's topological liveness analysis is an **interval graph**. Interval graphs are a subclass of **perfect graphs**, which means:

1. **Chromatic number = maximum clique size** (greedy coloring is optimal).
2. **Linear-time recognition and coloring** are possible (our Linear Scan is effectively this).
3. **No NP-hard subproblems** — unlike general graph coloring, interval graph coloring is polynomial.

### Perfect Graphs
**Definition**: A graph is *perfect* if for every induced subgraph, the chromatic number equals the maximum clique size. ([Wikipedia: Perfect graph](https://en.wikipedia.org/wiki/Perfect_graph))

**Key property**: In perfect graphs, the graph coloring problem, maximum clique problem, and maximum independent set problem are all solvable in polynomial time.

**Strong Perfect Graph Theorem** (Chudnovsky, Robertson, Seymour, Thomas 2006): A graph is perfect if and only if it contains no odd hole (odd cycle of length >= 5) and no odd antihole (complement of an odd cycle of length >= 5) as an induced subgraph.

**Perfect Graph Theorem** (Lovász 1972): The complement of a perfect graph is also perfect.

### Interval Graphs
**Definition**: An interval graph is the intersection graph of intervals on the real line. Two vertices are adjacent iff their intervals overlap. ([Wikipedia: Interval graph](https://en.wikipedia.org/wiki/Interval_graph))

**Why interval graphs are perfect**: Interval graphs are chordal (no induced cycles of length >= 4) and AT-free (no asteroidal triple). Every chordal graph is perfect. Therefore every interval graph is perfect.

**Characterization**: A graph is an interval graph if and only if:
1. It is chordal, AND
2. Its complement is a comparability graph.

**Greedy coloring is optimal**: Sort intervals by left endpoint, assign lowest available color. This matches our `color_linear_scan` implementation.

### Connection to TextureSynth's DAG
TextureSynth's node graph is a DAG (directed acyclic graph) processed in topological order. Each node output has a **live interval** [start_step, end_step] on this topological line. The interference graph built from these intervals is an interval graph because:

```
Node A output:  [step 2, step 5]  (defined at step 2, last read at step 5)
Node B output:  [step 3, step 7]  (defined at step 3, last read at step 7)
Node C output:  [step 6, step 8]  (defined at step 6, last read at step 8)

Interval graph:  A--B  (overlap [3,5])
                 B--C  (overlap [6,7])
                 A--C  (no overlap, no edge)
```

The chromatic number equals the maximum number of intervals overlapping at any single point (the maximum "register pressure" at any step). Linear scan achieves this minimum by the interval graph perfection guarantee.

**References**:
- [Perfect graph — Wikipedia](https://en.wikipedia.org/wiki/Perfect_graph)
- [Interval graph — Wikipedia](https://en.wikipedia.org/wiki/Interval_graph)
- [Goemans, Lecture 7: Perfect graphs (MIT 18.438)](https://math.mit.edu/~goemans/18438S14/lec7-perfect.pdf)

### SSA Form and Chordal Interference
In SSA-based compilers, phi nodes insert parallel copies at dominance frontiers. The interference graph of an SSA program is always chordal (no induced cycle >= 4). This is because SSA dominance constraints force the interference graph into a structure where every cycle of length >= 4 has a chord. Chordal graphs are perfect, so SSA-based register coloring is polynomial. TextureSynth's topological ordering is analogous: it imposes a total order on definitions and uses, producing interval (hence chordal, hence perfect) interference.

---

## 8. Code Example — Fused GLSL Output

Instead of:
```glsl
void main() {
    vec4 _local_0; // output of node 0
    vec4 _local_1; // output of node 1
    vec4 _local_2; // output of node 2
}
```
We generate (with register reuse):
```glsl
void main() {
    vec4 r0, r1;

    r0 = node_source(uv);                   // _local_0 -> r0
    r1 = node_step(uv, r0, param1);         // _local_1 -> r1
    r0 = node_step(uv, r1, param2);         // _local_2 -> r0 (reused!)

    imageStore(u_storage[nonuniformEXT(pc.out_storage_slots[0])], coord, r0);
}
```

---

## 9. Verification
- Standalone unit test `TestRegAlloc.cpp`: defines various graph topologies (linear, fan-out/fan-in, branches), runs liveness + interference + coloring, verifies no overlapping variables share a color and that color count matches theoretical minimum (chromatic number = max clique for interval graphs).

## 10. Child DOX Index
None — this is a leaf module with no subdirectories.
