# Graph-Fusion Activation Plan

## Goal
Wire `FusedGraphCompiler::compile()` into `Engine::set_graph()` to activate the entire graph-fusion pipeline. Fix cascading issues. Add tests.

## Steps

### Step 1: Swap compiler call in `set_graph()`
**File:** `src/engine/Engine.cpp`
- Add `#include "engine/graphfusion/FusedGraphCompiler.hpp"` to includes
- Line 405: Change `GraphCompiler::compile(ir_result.ir, node_lib_)` → `FusedGraphCompiler::compile(ir_result.ir, node_lib_, graph.output_node)`

### Step 2: Retire chain pipelines in `retire_all_passes_()`
**File:** `src/engine/EnginePassCompile.cpp` lines 211-219
- After retiring per-pass pipelines, also retire `chain_execs_` pipelines via `retired_passes_.push_back()`
- Clear `chain_execs_` and `chain_id_of_pass_`

### Step 3: Cache-hit path — skip chain-member passes
**File:** `src/engine/Engine.cpp` lines 480-488
- Read `chain_index_of_pass` from the plan
- Skip passes where `chain_index_of_pass[i] != UINT32_MAX` in per-node cache check
- Add separate fused-key cache check loop for `plan.chains`

### Step 4: Cache-hit path — skip pipeline creation for chain members
**File:** `src/engine/Engine.cpp` lines 504-514
- Gate `create_pass_pipeline_()` on `chain_index_of_pass[i] == UINT32_MAX`
- Chain member passes still get `assign_bindless_slots_()` (for input binding)

### Step 5: Async path — set `chain_member`, skip futures
**File:** `src/engine/Engine.cpp` lines 526-546
- Set `pp.chain_member = (chain_index_of_pass[i] != UINT32_MAX)` 
- Skip `compile_compute_async()` for chain members

**File:** `src/engine/EnginePassCompile.cpp` line 112
- Add `|| pp.chain_member` to the readiness skip condition

### Step 6: Fix SSBO seed loop — add float-input defaults
**File:** `src/engine/EngineParams.cpp` lines 29-36
- After params loop, add float-input default seeding: iterate `type->inputs` where `SocketType::Float`, write `inp.default_value` at `base + type->params.size() + float_input_idx`

### Step 7: Remove dead fields
- `src/engine/PassPlan.hpp:35-36`: Remove `mutable bool dirty` and `mutable uint64_t last_executed_gen`
- `src/engine/GraphCompiler.hpp:18`: Remove `std::string glsl;`

### Step 8: Integration tests + verify existing tests
- New file `tests/test_fused_graph_activation.cpp`:
  1. `FusedActivation_ProducesChains` — verify chains populated
  2. `FusedActivation_ChainMembersSkipPipelineCreation` — chain members have empty shader_glsl
  3. `FusedActivation_SetGraphActivatesChains` — full Engine API, verify chain_execs_
  4. `FusedActivation_ParamUpdateNoRecompile` — param tweak fires 1 chain dispatch
  5. `FusedActivationSetActiveNode` — rebuilds chains for shorter path
- Verify `ChainDispatch_OneDispatchPerChain`, `ChainDispatch_ParamUpdateFiresOneChain`, `ChainDispatch_BypassedChainClearsMembers` now pass

### Step 9: Final build + full test suite
- `cmake --build build --config Release --target engine_tests`
- Run all tests, expect 132/132 or at minimum the 4 previously-failing chain tests now pass

## Key Files
- `src/engine/Engine.cpp` — Steps 1, 3, 4, 5
- `src/engine/EnginePassCompile.cpp` — Steps 2, 5
- `src/engine/EngineParams.cpp` — Step 6
- `src/engine/PassPlan.hpp` — Step 7
- `src/engine/GraphCompiler.hpp` — Step 7
- `tests/test_fused_graph_activation.cpp` — Step 8 (new)
- `CMakeLists.txt` — Step 8 (add test file)
