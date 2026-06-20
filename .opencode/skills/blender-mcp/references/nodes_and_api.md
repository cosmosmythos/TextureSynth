# TextureSynth live-test API reference

Canonical node inventory + the Python calls you need to build graphs inside a running Blender via MCP. Read this before constructing a graph from scratch.

## MCP execution model (why the patterns below look the way they do)

The Blender MCP addon (`%APPDATA%\Blender Foundation\Blender\5.0\scripts\addons\blender_mcp_addon.py`) only exposes `execute_code`, which calls `exec(code, {"bpy": bpy})`:
- **Fresh namespace per call** — re-import everything each time. Use the `_ts_pkg()` resolver (see `scripts/_resolver.py`).
- **Runs on the main thread** via `bpy.app.timers.register` — `bpy.ops` is safe.
- **`bpy.context` is the user's real editor state** — `context.active_node` is whatever they clicked. Setting `tree.nodes.active` in code does NOT change it for operator calls. Use `bpy.context.temp_override(...)` **inside an editor area context**, or prefer the direct socket/node API.
- **Stdout is the only return channel** — `print()` everything. Structure output as a report.

## Node inventory

Two node families. Always confirm a node exists at runtime with `engine.node_library().all()` or by checking `bpy.types` — the manifest list grows.

### Factory-generated nodes

Class id pattern: `TS_<SvTypeCapitalized>_Node`. Generated from `shader_assets/nodes/*.node.json` at addon load by `ADDON/nodes/factory.py`. Current set (from manifests):

| sv_type | bl_idname | Category | Has params | Inputs |
|---|---|---|---|---|
| `perlin`        | `TS_Perlin_Node`        | NOISE  | yes (period, octaves, lacunarity, roughness, speed, seed) | none |
| `simplex`       | `TS_Simplex_Node`       | NOISE  | yes | none |
| `worley`        | `TS_Worley_Node`        | NOISE  | yes | none |
| `gabor`         | `TS_Gabor_Node`         | NOISE  | yes | none |
| `value`         | `TS_Value_Node`         | NOISE  | yes | none |
| `white_noise`   | `TS_White_Noise_Node`   | NOISE  | yes | none |
| `invert`        | `TS_Invert_Node`        | FILTER | none | `mask` (float, default 1.0), `color` (vec4) |
| `grayscale`     | `TS_Grayscale_Node`     | FILTER | `mode` (int enum) | `mask`, `color` |
| `combine_rgba`  | `TS_Combine_Rgba_Node`  | COLOR  | — | R, G, B, A |
| `separate_rgba` | `TS_Separate_Rgba_Node` | COLOR  | — | `color` (vec4) |
| `shuffle`       | `TS_Shuffle_Node`       | CHANNEL | `r_src`/`g_src`/`b_src`/`a_src` enum | `color` (vec4) |

The 6 noise types + `color_const` are the only nodes that expose `format_override` (Mono/UV/RGB/RGBA/ID). See `factory._FORMAT_OVERRIDE_SV_TYPES`.

### Specialized (hand-written) nodes

Class ids are literal, not pattern-derived. Defined under `ADDON/nodes/specialized/`.

| sv_type | bl_idname | Category | Notes |
|---|---|---|---|
| `color_const` | `TS_Color_Const_Node` | INPUT | Solid color source. Format-overrideable. |
| `blend`       | `TS_Blend_Node`       | BLEND | 3 inputs: `mask`, `a`, `b`. `mode` enum → 0..28 int. Format NOT overrideable. |
| `image`       | `TS_Image_Node`       | INPUT | Pixels uploaded from a `bpy.types.Image` via `engine_bridge.upload_node_image`. |
| `levels`      | `TS_Levels_Node`      | CHANNEL | 1 input, 1 output. 25 params (`in_low_l`..`out_high_a` + `channel_mode` enum). |
| — (marker)    | `TS_Output_Node`      | OUTPUT | Bake sink. Not an sv_type (`sv_type=None`). Inputs are dynamic bake targets. |

## Engine bridge API (the part pytest can't reach)

All under `ADDON/core/engine_bridge.py`. These are the functions a live test calls directly when it wants determinism instead of waiting on the timer.

| Function | Purpose |
|---|---|
| `_find_node_tree()` | Resolve the active editor's `TextureSynthTreeType` tree (fallback: first in `bpy.data.node_groups`). Returns `None` if none. |
| `submit_graph()` | Build the engine `Graph` from the tree, push it, return generation (0 = failed; check `engine.last_error()`). |
| `update_params_only(force_submit=True)` | Re-dispatch params without rebuilding topology. Returns `'landed'`/`'in_flight'`/`'idle'`/`'invalid'`. |
| `sync_node_errors(tree)` | Pull compile errors, paint failed nodes red, populate `node.ts_compile_error`. |
| `upload_node_image(node)` | Push a `TS_Image_Node`'s `bpy.types.Image` pixels to Vulkan. Called automatically on submit. |
| `_active_subgraph_fingerprint(tree)` | Stable hash of topology + mute state. The timer compares this to detect missed changes. |

Module-level state that bites tests: `_submitted_generation`, `_last_active_fingerprint`, `_last_pushed_param_hash`, `_last_mute_snapshot`. If a test manipulates the graph behind the bridge's back, these can go stale and the timer won't re-dispatch. Call `evaluation.request_topology_update()` to force a clean re-submit.

## Evaluation trigger API

`ADDON/core/evaluation.py` — what the UI/operators call to wake the timer.

| Function | When to call |
|---|---|
| `request_topology_update()` | After adding/removing nodes or links, changing mute, changing format_override. Triggers full re-submit. |
| `request_param_update()` | After changing only a slider/enum value. Triggers param re-dispatch (no graph rebuild). |

The timer (`_evaluation_timer`, ~50-100ms) drains these flags. In a test, either set the flag and sleep ~0.3s, or call `engine_bridge.submit_graph()` / `update_params_only(force_submit=True)` directly for synchronous behavior.

## Engine object API (`cpp_module.get_engine()`)

Mirrors `texturesynth_core`. The methods you'll touch in tests:

| Method | Returns |
|---|---|
| `set_graph(graph)` | `int` generation; `0` on validation error |
| `is_generation_ready(gen)` | `bool` — shader compile + pipeline ready |
| `poll_pending_compiles()` | `None` — drains the async compile fence queue |
| `submit_render(pc, gen)` | ticket `int` |
| `poll_readback()` | `(pixels_np_hxw4, gen)` or `None` if no frame ready |
| `async_in_flight()` | `bool` |
| `last_error()` | `str` |
| `last_error_record()` | `EngineError` (`.code`, `.message`, `.failed_node`, `.phase`, `.is_error()`) |
| `failed_node()` | stable_id of the node that failed compile |
| `node_library().all()` | `{sv_type: NodeType}` — authoritative node list |
| `set_resolution(w, h)` / `set_precision(0|1|2)` | scene resolution / R8/R16/R32 |
| `bake()` | list of `{name, width, height, pixels}` dicts (called by `texturesynth.bake`) |

## Graph building — the gotchas

### bl_idname must be exact
```python
# Factory nodes: TS_<Capitalized>_Node
perlin = tree.nodes.new('TS_Perlin_Node')
# But:
white = tree.nodes.new('TS_White_Noise_Node')   # underscores preserved from sv_type
# Specialized nodes: hand-written ids, no pattern
blend = tree.nodes.new('TS_Blend_Node')
out   = tree.nodes.new('TS_Output_Node')
```
When unsure, list registered TS classes:
```python
[k for k in dir(bpy.types) if k.startswith('TS_') and k.endswith('_Node')]
```

### Wiring links
```python
# engine_bridge uses stable_id(); node must exist (init assigns UUID) before link.
tree.links.new(perlin.outputs[0], invert.inputs['color'])
```
Socket access by index works for single-output nodes; by name is safer for multi-input nodes (blend, combine_rgba).

### as_socket param slot ordering (root ADDON/core/engine_bridge.py:328)
When a manifest param has `"as_socket": true`, it becomes an input socket that sits **after** all regular inputs in the engine's SSBO layout. If you wire a link and the engine reports a missing/wrong input, the bridge reorders with:
- regular inputs first (sequential)
- as_socket inputs after, in socket order

You rarely hit this unless you're testing a node with `as_socket` params. `invert`/`blend`/`grayscale` have a `mask` float input that is a *regular* input (default 1.0), not an as_socket param — `mask` is socket 0.

### Format override
Only noise + color_const nodes respect `format_override`. Setting it on others is a no-op. Changing it triggers a topology update (socket types rebuild via `rebuild_output_sockets`).

### Output node targets
`TS_Output_Node` starts with one input socket named "Base Color". The socket **name** becomes the baked image data-block name.

**Under MCP, prefer the direct socket API** over the operators — `texturesynth.output_target_add` reads `context.active_node`, which is the user's real selection (not something you can set reliably in a headless `execute_code` run):
```python
# Direct (MCP-safe):
out.inputs[0].name = "Albedo"
out.inputs.new('TS_BakeTargetSocketType', "Roughness")   # add a target
out.inputs.remove(out.inputs[-1])                        # remove last target
```
The operator path (`bpy.ops.texturesynth.output_target_add()`) only works when wrapped in a `temp_override(area=<NODE_EDITOR area>, active_node=out)` — fiddly, so the scripts avoid it. `texturesynth.bake` itself does NOT need `context.active_node` (it resolves the Output node via `engine_bridge._active_output_node`), so it runs fine under MCP.

## Reading the user's graph (Mode A introspection — read-only)

Before mutating anything the user has open, dump it. This is what `scripts/introspect.py` does. The shape:

```python
tree = engine_bridge._find_node_tree()
for n in tree.nodes:
    sv = getattr(n, 'sv_type', None)          # None for the Output marker node
    sid = n.stable_id() if hasattr(n, 'stable_id') else None
    mute = bool(getattr(n, 'mute', False))
    ann = getattr(type(n), '__annotations__', {})
    params = {k: getattr(n, k) for k in ann
              if k not in ('ts_uuid','ts_compile_error','format_override')}
    err = getattr(n, 'ts_compile_error', '')  # non-empty => this node failed compile
```

Link tracing (who feeds whom):
```python
for link in tree.links:
    if not link.is_valid:
        continue
    from_idx = list(link.from_node.outputs).index(link.from_socket)
    to_idx   = list(link.to_node.inputs).index(link.to_socket)
    print(f"{link.from_node.name}[{from_idx}] -> {link.to_node.name}[{to_idx}]")
```

Engine state to read alongside:
```python
engine = cpp_module.get_engine()
gen   = engine_bridge._submitted_generation
ready = engine.is_generation_ready(gen) if gen else False
rec   = engine.last_error_record()        # .code, .phase, .message, .failed_node
```

The introspection output drives the **quiz** (SKILL.md "Mode A"): state back to the user what you see and confirm the task before changing anything. If the graph has ≥3 nodes or any links, the quiz is mandatory — guessing wrong wastes their setup.

## Reading pixels back

After a dispatch lands, two sources:

1. **`bpy.data.images["TS_Preview2D"]`** — what the user sees. Float RGBA, `W*H*4` flat in `.pixels`. Dimensions match scene resolution.
2. **`engine.poll_readback()`** — direct from Vulkan, bypasses the blit. Returns `(numpy[h,w,4], gen)`.

For statistical assertions (mean, non-zero fraction), use numpy on the poll_readback array — it's already shaped and avoids a Blender image copy.

## Error inspection

| Symptom | Look at |
|---|---|
| `set_graph` returned 0 | `engine.last_error_record()` — `.code` (GraphValidation/GraphCompile), `.failed_node`, `.phase` |
| A node turned red in UI | `node.ts_compile_error` string + `engine.failed_node()` stable_id |
| `TS_Preview2D` is all black | `engine_bridge._invalidate_output_image()` ran — a submit failed. Check `engine.last_error()`. |
| No dispatch after slider change | `_last_pushed_param_hash` matched the signature (no-op). Force via `request_param_update()` or `update_params_only(force_submit=True)`. |

`EngineErrorCode` is an enum with a `None` member — in Python use `tc.EngineErrorCode['None']` (root §5).
