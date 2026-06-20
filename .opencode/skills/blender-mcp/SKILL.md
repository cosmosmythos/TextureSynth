---
name: blender-mcp
description: Use when testing the TextureSynth Blender addon inside a LIVE Blender process via Blender MCP (the `blender_mcp_addon.py` socket server on port 9876). Two modes — (1) build/sweep/bake tests against a fresh graph, and (2) introspect + verify whatever graph the user ALREADY has open, quizzing them or yourself about the task before mutating anything. Use whenever the user says "test in Blender", "does the addon work", "verify the node", "check the graph builds", "sweep the slider", "bake and inspect", "what's in my graph", "test what I have open", "check my setup", or wants to confirm an addon change behaves end-to-end in the real Blender environment. Also use to diagnose addon/engine-bridge bugs that only reproduce with a live bpy.context and running Vulkan engine.

---

## What I do

Drive the **installed TextureSynth extension** from inside a running Blender through the Blender MCP socket server. The addon is event-driven (`ADDON/core/evaluation.py` runs a persistent timer + msgbus), so a test is not a function call — it is: *change graph state, let the timer dispatch, then read back the result.*

The scripts under `scripts/` are canonical, copy-paste-safe patterns. Read the one that matches the test before writing your own — the graph-building API has several non-obvious contracts (stable IDs, socket format overrides, `as_socket` param slot ordering) that are easy to get wrong.

## The Blender MCP contract (read first — this shapes every script)

The MCP addon is at `%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\blender_mcp_addon.py`. It exposes **one** useful command type: `execute_code`. Everything else (`get_scene_info`, etc.) is viewport/object-only and useless for node trees. Key facts:

| Fact | Implication |
|---|---|
| `execute_code` runs `exec(code, {"bpy": bpy})` — **fresh namespace per call** | Every script must self-contain its imports. Nothing persists between calls. |
| Runs on the **main thread** via `bpy.app.timers.register(..., first_interval=0.0)` | `bpy.ops.*` is valid. But context is the user's *real* editor state. |
| **Stdout is captured** via `redirect_stdout` and returned as the result | `print()` is the **only** output channel. Structure output as a report. No return values. |
| `bpy.context` reflects whatever the user has selected right now | `context.active_node` is NOT something you set by assigning `tree.nodes.active` in a headless run — it's the user's click. For operators needing it, use `bpy.context.temp_override` or direct socket API. |
| Server on `localhost:9876` | User must click "Connect to Claude" in the BlenderMCP panel first. If a call errors with connection refused, tell them to connect. |

### How to resolve the addon package

The extension's import path depends on how it's loaded. Prefer the resolver helper and **never hardcode** `import texturesynth`:

```python
import bpy, addon_utils, importlib, sys
def _ts_pkg():
    # Try the bl_ext path (4.2+ extension), then fall back to legacy addon name.
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        if name in sys.modules:
            return importlib.import_module(name)
    # Not imported yet: force-load via the registered module.
    for mod in sys.modules.values():
        core = getattr(mod, "core", None)
        if core and hasattr(core, "cpp_module"):
            return mod
    raise RuntimeError("TextureSynth addon not loaded in this Blender")
```

Every bundled script begins with this resolver. If it raises, the extension isn't enabled — tell the user to enable it in Preferences → Extensions.

## The addon test loop

Every live test follows five steps. Run them as separate `execute_code` calls so you can inspect intermediate output.

| Step | What | Key call |
|---|---|---|
| 1. Find tree | Get the active TextureSynth node tree | `engine_bridge._find_node_tree()` |
| 2. Build (or read) graph | Create nodes/links OR inspect what's already there | `tree.nodes.new(bl_idname)` + `tree.links.new(...)` |
| 3. Trigger dispatch | Ask the eval loop to submit | `evaluation.request_topology_update()` (topology) or `request_param_update()` (slider) |
| 4. Wait for GPU | Async compile + dispatch is non-blocking | poll `engine.is_generation_ready(gen)` + `engine.poll_readback()` |
| 5. Read back | Pixels land in `bpy.data.images["TS_Preview2D"]` | `img.pixels[:]` or `engine.poll_readback()` |

Steps 3-5 normally happen automatically in `_evaluation_timer` (~50-100ms). In a test you can either **let the timer run** (realistic, slower) or **drive the bridge directly** via `engine_bridge.submit_graph()` + `update_params_only(force_submit=True)` (deterministic, faster).

## Two testing modes

### Mode A — Verify what's already open (QUIZ FIRST)

The user often has a graph already built and wants you to test *that*, not a fresh one. **Before mutating anything**, introspect and confirm the task. This avoids destroying their work.

1. Run `scripts/introspect.py` — prints every node, its sv_type, params, links, and engine state. **Read-only.** Safe on any graph.
2. **Quiz the task.** Either ask the user, or pose the question to yourself and answer it from the introspection output. Examples:
   - "I see a Perlin → Levels → Output graph, 1024², R16. You said 'test roughness' — do you mean sweep the Perlin `roughness` param and check pixels change? Or add a roughness bake target?"
   - "The graph has no Output node wired. Did you intend to bake, or just preview?"
3. **State the plan in one sentence** and only then touch state. If a step would mutate the user's graph (add/remove nodes, rewire, change sliders), say so explicitly and prefer non-destructive variants: add a *new* Output target instead of renaming theirs; sweep a param and restore it after; snapshot node positions before re-locating.
4. Run the matching Mode B script (adapted) or a custom check, then report.

The quiz is mandatory when the introspection shows a non-trivial graph (≥3 nodes or any links) — guessing wrong wastes the user's setup.

### Mode B — Build a fresh test graph

When there's nothing useful open, or the test needs a controlled setup, build from scratch. Clean the tree first (only after confirming with the user it's not their work). Use the scripts below as scaffolds.

## When to use me

- After editing anything in `ADDON/` and deploying (via the `blender-addon` skill) — confirm the change didn't break graph build / dispatch / readback.
- Verifying a new node behaves: appears in the Add menu, generates params, accepts links, compiles, produces non-zero output.
- Sweeping a slider to confirm the engine re-dispatches and pixels change.
- "Check my setup" / "test what I have open" — Mode A. Introspect, quiz, then verify.
- Reproducing a user-reported bug that needs a real `bpy.context` (active node, editor space, msgbus) the pytest suite can't construct.
- Bake smoke test: Output node targets → `texturesynth.bake` → image data-blocks exist with non-zero pixels.

## When NOT to use me

- Testing the C++ engine or `texturesynth_core` binding **without** Blender — use `pytest tests/python/` (root §6). Faster, no GUI.
- C++ engine internals — use the gtest suite (`-DBUILD_TESTS:BOOL=ON`).
- Anything expressible in `pytest`. Prefer `pytest` when it fits.

## Bundled scripts (read the relevant one before writing new code)

| Script | Mode | Use when |
|---|---|---|
| `scripts/_resolver.py` | both | Helper to find the addon package. Pasted at the top of every other script — not run alone. |
| `scripts/smoke_test.py` | B | First-run / "does the addon load at all". Engine live, node library populated, tree exists. |
| `scripts/introspect.py` | A | **Read-only** dump of the current graph: nodes, params, links, engine state, last error. Always run this first in Mode A. |
| `scripts/build_graph.py` | B | Fresh Perlin → Invert → Output. Template for wiring links + setting props. |
| `scripts/param_sweep.py` | B | Confirm a slider change re-dispatches and readback pixels change. |
| `scripts/bake_targets.py` | B | End-to-end bake: add Output targets, call `texturesynth.bake`, verify image data-blocks + non-zero pixels. |

Adapt, don't copy verbatim — these are scaffolds showing correct API calls and ordering, not assertions locked to specific pixel values.

To run one: send its full text via the MCP `execute_code` tool. Output comes back as captured stdout (the scripts use `print` for exactly this reason).

## API reference

Full node list, socket layouts, and bl_idnames live in `references/nodes_and_api.md`. **Read it before constructing a graph from scratch.** It documents the two easiest things to get wrong:

1. **Node `bl_idname`s** are not uniform. Factory-generated nodes use `TS_<SvTypeCapitalized>_Node` (e.g. `TS_Perlin_Node`); specialized nodes use hand-written ids (`TS_Blend_Node`, `TS_Image_Node`, `TS_Levels_Node`, `TS_Color_Const_Node`, `TS_Output_Node`).
2. **`as_socket` params become input sockets**, and the engine's slot ordering is "regular inputs first, then as_socket inputs" (`ADDON/core/engine_bridge.py:328-341`).

## Common live-test pitfalls

- **Operators needing `context.active_node`.** In MCP, `context` is the user's real selection — setting `tree.nodes.active` in code does NOT change `context.active_node` for the operator call. Use `bpy.context.temp_override(active_node=node, ...)` **inside the editor area context**, or skip the operator and use the direct API (e.g. `node.inputs.new(...)` instead of `bpy.ops.texturesynth.output_target_add()`). The scripts prefer the direct API for this reason.
- **`tree.nodes.new` positioning.** Headless MCP runs stack nodes at (0,0). Set `node.location = (x, y)` only if layout matters (it usually doesn't for tests).
- **Every TS node needs a `stable_id()`** before the engine sees it — `init` assigns the UUID. Create-then-link and create-then-submit are both fine.
- **Mute rewiring searches ALL inputs**, not input[0] (root §5 gotcha). For filter nodes socket 0 is often the mask; data is socket 1.
- **One VkInstance per process** (root §5). Never call `cpp_module.shutdown()` mid-test and expect the addon to keep working. Tests leave the engine running.
- **Readback is async.** `submit_graph()` returning non-zero means *accepted*, not *rendered*. Poll `is_generation_ready(gen)` then `poll_readback()` (returns `None` until a frame lands).
- **`TS_Preview2D`** is the live preview image. Black after a submit = `engine_bridge._invalidate_output_image()` ran = submit failed. Check `engine.last_error()` for the cause; black is the symptom, not the diagnosis.

## Reporting results

After a test run, report concretely:
- Which script ran and whether each step's assertion passed (quote the printed PASS/FAIL lines).
- Engine generation number, `last_error()` string (or "None"), and any compile error on a node (`node.ts_compile_error`).
- For readback/bake: image dimensions and a pixel statistic (mean / non-zero fraction), not "it worked".
- If a step failed, quote the actual `engine.last_error()` / Python traceback — don't paraphrase.

Do not claim a change is verified unless readback actually produced non-trivial pixels. For Mode A, also state what the quiz concluded and which (if any) mutation you performed.
