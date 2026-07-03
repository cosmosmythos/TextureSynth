# Mute Bug Fix Plan

Three root causes, three fixes. No band-aids.

---

## Fix 1: Engine.cpp passes wrong output_node to compiler (CRITICAL)

**Root cause:** `Engine.cpp:406` passes `graph.output_node` (raw, potentially muted ID) instead of `ir_result.ir.output_node` (corrected during validation).

**Effect:** Muted node ID not found in eval_order → empty path → all passes get `shader_glsl = ""` → shaderc error `#version: compute shaders require...` → red node.

**Fix:**
- `src/engine/Engine.cpp:406` — change `graph.output_node` to `ir_result.ir.output_node`

**Also defensive fix:**
- `src/engine/graphfusion/FusedGraphCompiler.cpp:116` — when `path.nodes.empty()`, the early return at line 132 skips per-node GLSL emission at lines 222-235, leaving all `shader_glsl = ""`. The fallback path must emit per-node GLSL before returning.

---

## Fix 2: sync_node_errors infinite feedback loop (FLICKERING)

**Root cause:** `sync_node_errors()` writes `node.use_custom_color = False` on every non-failed node every timer tick, even when already False. Blender RNA fires `tree.update()` → `request_topology_update()` → `submit_graph()` → `sync_node_errors()` → loop at 100Hz.

**Effect:** Viewport flickers, GPU/CPU overloaded, timer locked at 0.01s.

**Fix:**
- `ADDON/core/engine_bridge.py:sync_node_errors()` — guard `use_custom_color` writes: only write when the value actually needs to change. Track previous state or check `if getattr(node, 'use_custom_color', False): node.use_custom_color = False`. Same for `ts_compile_error`.
- Also: `submit_graph()` at line 608 calls `sync_node_errors()` AGAIN after the graph is already submitted. This is redundant with the timer's call at evaluation.py:114. Remove the call from `submit_graph()` OR the timer — not both.

---

## Fix 3: Muted active node → black image, not error (ARCHITECTURE)

**Root cause:** When the active node is muted, the Python addon sets `output_src_id` to the muted node's ID. The C++ engine should handle this gracefully (and now will after Fix 1), but the Python side should also be defensive.

**Fix (Python side):**
- `ADDON/core/engine_bridge.py:_build_graph_and_params()` — when setting `output_src_id`, skip muted nodes. If the active node is muted, walk its input chain to find the first non-muted upstream node, matching what `GraphIR.cpp:254-265` does.

---

## Test Gap: No compile-pipeline tests for mute

**Why tests didn't catch it:** All 7 mute tests in `test_graph_validation.cpp` test `validate_graph()` only. They verify IR correctness (nodes excluded, rewired, output redirected). But NONE pass the IR to `FusedGraphCompiler::compile()` or call `Engine::set_graph()`. The bug lives in the consumption of validation output, not in validation itself.

**Fix:**
- Add `test_graph_validation.cpp` tests that call `FusedGraphCompiler::compile()` with a muted output node and verify:
  - `result.success == true`
  - All passes have non-empty `shader_glsl`
  - `ir.output_node` is used, not the muted node ID
- Add `test_full_pipeline.cpp` integration test: build graph with muted node → `set_graph()` → verify no compile error, pipeline created.

---

## Order of execution

1. **Fix 1** (Engine.cpp:406) — unblocks everything, no more `#version` error on mute
2. **Fix 2** (sync_node_errors guard) — stops the flickering
3. **Fix 3** (Python defensive output resolution) — future-proof
4. **Tests** — prevent regression
