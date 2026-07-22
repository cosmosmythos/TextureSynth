---
name: blender-mcp
description: Run Python code in live Blender via TCP socket server. Query scenes, node trees, engine state.
compatibility: opencode
---

## What I do

Execute arbitrary Python in Blender's main thread via TCP MCP server. Read scene data, node trees, engine state. No Blender restart needed.

## Protocol

| Item | Value |
|---|---|
| Host:Port | `127.0.0.1:9876` |
| Encoding | JSON, null-byte (`\0`) delimited |
| Request | `{"type":"execute","code":"...","strict_json":true}` |
| Response | `{"status":"ok"/"error","result":{...},"message":"..."}` |
| Server | `%APPDATA%\Blender Foundation\Blender\5.2\extensions\lab_blender_org\mcp\mcp_to_blender_server.py` |

## Quick usage

```powershell
# Inline query
python .opencode/skills/blender-mcp/scripts/mcp_fetch.py "import bpy; result={'ver':bpy.app.version_string}"

# From script file
python .opencode/skills/blender-mcp/scripts/mcp_fetch.py my_script.py

# Default: scene overview
python .opencode/skills/blender-mcp/scripts/mcp_fetch.py

# Run any debug script:
python .opencode/skills/blender-mcp/scripts/engine_state.py
python .opencode/skills/blender-mcp/scripts/error_diag.py
```

## Scripts

| Script | Run | What you get |
|---|---|---|
| `mcp_fetch.py` | `python mcp_fetch.py "code"` | Core helper — send code, get JSON. Importable from other scripts. |
| `dissect_scene.py` | `python dissect_scene.py` | All 20 nodes, 21 links, sockets, UUIDs, engine status |
| `engine_state.py` | `python engine_state.py` | Pipeline ready? Generations (compile/installed/revision), VRAM bytes, VMA heap telemetry (11 fields) |
| `graph_inspect.py` | `python graph_inspect.py` | Compiled graph output node ID, param layout (node_id→offset), float count |
| `error_diag.py` | `python error_diag.py` | `last_error_record()` (code/message/failed_node/phase) + per-node `ts_compile_error` from tree |
| `perf_timings.py` | `python perf_timings.py` | Per-pass GPU timings (pass_index, duration_us, available), gen bump, readback in-flight |
| `node_params.py` | `python node_params.py` | Every node's sliders, enums, depth/format — safe against bpy_prop_array |
| `shader_cache.py` | `python shader_cache.py` | All SPIR-V keys: fused chains (`blur→invert`), epoch, external_socket_masks, param_socket_masks |
| `enable_cpp_logging.py` | `python enable_cpp_logging.py` | Hook C++ `set_log_callback(print)` + bump addon prefs to DEBUG. C++ logs appear in Blender system console. |
| `disable_cpp_logging.py` | `python disable_cpp_logging.py` | Remove C++ log callback |
| `ts_quickref.py` | `python ts_quickref.py` | Copy-paste constants for common MCP queries |
| `trace_timer.py` | `python trace_timer.py` | Read evaluation-timer state + dispatch decision chain from MCP (no addon edits). Identifies blocker: `generation_not_ready`, `compiling_not_ready`, `nothing_dirty`, `no_pipeline`, `engine_not_loaded` |
| `full_pipeline.py` | `python full_pipeline.py` | Full round-trip: set_resolution → add_node → set_graph → compile → params → submit → readback |

All paths relative to `scripts/`. All scripts handle engine-not-loaded case gracefully.

## Confirmed engine API calls (C++ → Python bindings)

These call `get_engine()` → `Engine` object:

| Call | Returns | Debug use |
|---|---|---|
| `e.has_pipeline()` | bool | Pipeline compiled and ready |
| `e.is_ready()` | bool | Engine in Ready state |
| `e.engine_state()` | int | 0=Idle, 1=Building, 2=Ready, 3=Error |
| `e.current_revision()` | uint64 | Latest submitted generation |
| `e.compile_generation()` | uint64 | Generation being compiled |
| `e.installed_generation()` | uint64 | Generation ready to dispatch |
| `e.is_generation_ready(gen)` | bool | Specific gen finished compiling |
| `e.async_in_flight()` | bool | Render job pending readback |
| `e.resource_count()` | int | Live VRAM image count |
| `e.resource_bytes()` | uint64 | VRAM bytes used |
| `e.resource_budget_bytes()` | uint64 | VRAM budget limit |
| `e.get_vma_stats()` | dict | Full VMA: node counts, heap stats, aliasing efficiency, GPU pressure |
| `e.last_pass_timings` | list[PassTiming] | Each: `p.pass_index`, `p.duration_us`, `p.available` |
| `e.last_error()` | str | UTF-8 error message (empty if no error) |
| `e.failed_node()` | uint64 | Node ID that caused compile failure (0 = none) |
| `e.last_error_record()` | EngineError | `.code`, `.message`, `.failed_node`, `.phase`, `.graph_generation`, `.is_error()` |
| `e.current_graph()` | Graph | C++ graph ref — use `.output_node` for node ID |
| `e.param_layout()` | dict | `{node_id: base_offset}` in the GPU SSBO |
| `e.total_param_floats()` | int | Total float count in param buffer |
| `e.precision()` | int | Precision mode |
| `e.graph_default_depth()` | BitDepth | Default bit depth (cast to str for JSON) |
| `e.update_node_params_by_name(id, dict)` | — | Push per-node params to GPU |

### Graph building API (for full round-trips)

| Call | Signature |
|---|---|
| `tsc.Graph()` | Construct empty graph |
| `g.add_node(id, type, format_override, debug_name, muted, bypassed, depth_mode, absolute_depth)` | Add node. Only `id` (uint64) and `type` (str) required. Defaults: `RGBA, "", False, False, Auto, F16` |
| `g.add_connection(src_id, src_sock_idx, dst_id, dst_sock_idx)` | Wire nodes by integer socket index (0-based) |
| `g.set_output(node_id)` | Set active output node |
| `g.add_output_target(node_id, name, socket_idx=0)` | Named bake target |
| `g.output_node` (prop) | Read current output node ID |
| `tsc.PushConstants()` | Construct dispatch constants. Set `.resolution_x`, `.resolution_y`, `.seed` (uint32, pass int not float), `.time` |
| `e.set_resolution(w, h)` | Must call BEFORE `set_graph()` |
| `e.set_graph(graph)` → uint64 generation | Submit for async compile. 0 = failure |
| `e.poll_pending_compiles()` | Drain async compile queue (call each tick) |
| `e.is_generation_ready(gen)` → bool | Check if submit→compile finished |
| `e.update_node_params_by_name(node_id, {"name": val})` | Push slider/enum values to GPU SSBO |
| `e.update_node_params_by_id(node_id, [float, ...])` | Push params by flat index order |
| `e.submit_render(pc, gen)` → uint64 ticket | Dispatch GPU render. 0 = ring full |
| `e.readback_sync()` → numpy[H,W,4] float32 RGBA | Block until GPU done, get pixels |
| `e.poll_readback()` → (numpy, gen) or None | Non-blocking async readback |
| `e.bake()` → list of `{name, width, height, pixels}` | Bake all output targets |
| `e.upload_image(node_id, numpy[H,W,4], w, h)` → bool | Upload pixel data for image/texture nodes |

`texturesynth_core` module-level:

| Call | Debug use |
|---|---|
| `texturesynth_core.set_log_callback(fn)` | Stream C++ engine logs to Python. Set to `None` to disable. Callback: `fn(level_str, msg_str)`. |

## Blender 5.2 confirmed APIs for TextureSynth

| What to use | Gets you |
|---|---|
| `bpy.context.preferences.addons["bl_ext.user_default.texturesynth"]` | Check if TS extension is enabled (raises KeyError if not) |
| `bpy.context.preferences.addons` | Enabled addons list (`bl_ext.user_default.texturesynth`) |
| `t.name.startswith("TextureSynth")` | Find TS node tree |
| `t.bl_idname` | `"TextureSynthTreeType"` |
| `type(t).__name__` | `"TextureSynthTree"` |
| `n.bl_rna.identifier` | Node type: `"TS_Invert_Node"`, `"TS_Blur_Node"`, etc. |
| `n.items()` | Custom ID properties (TS nodes store params as registered RNA props, not ID props — use `bl_rna.properties` instead) |
| `getattr(n, p.identifier)` | Read node param by RNA property identifier |

## Import rules (verified in Blender 5.2)

MCP code runs inside Blender's Python where the addon is fully loaded. All these work:

```python
# Get engine — use either:
from bl_ext.user_default.texturesynth.core.cpp_module import get_engine
e = get_engine()

# OR (slightly more robust in edge cases):
import sys
e = sys.modules["bl_ext.user_default.texturesynth.core.cpp_module"].get_engine()
```

```python
# Module-level C++ API (Graph, PushConstants, set_log_callback):
import texturesynth_core as tsc
g = tsc.Graph()
pc = tsc.PushConstants()
tsc.set_log_callback(fn)
```

**What does NOT work:**
- `texturesynth_core.get_engine()` — `get_engine()` is in `cpp_module`, not at module level

**One gotcha:** `engine_bridge` has both a function `submitted_generation()` and a variable `_submitted_generation`. Use `eb.submitted_generation()` for the function, `eb._submitted_generation` for the raw variable.

## Serialization rules (must follow exactly)

- **`result` must be a dict.** Always: `result = {"key": value}`.
- **C++ binding objects** (`EngineError`, `PassTiming`, `BitDepth`, `Graph`, `NodeType`) are NOT subscriptable. Access as `.attribute` not `["attr"]`.
- **bpy_prop_array / Vector / Color / Matrix** are NOT JSON-serializable. Convert: `round(float(val), 6)`, `int(val)`, `bool(val)`, `str(val)`.
- **Enum objects** from C++ bindings (`BitDepth.F16`) — cast to `str()`.
- **MCP request code** must be a single string passed as CLI arg or file. Multi-line is fine inside quotes.

## Script structure rule (critical — agents keep getting this wrong)

**Never use PowerShell here-strings (`@"..."@`) for Python multi-line code.**

PowerShell 5.1 here-strings have fragile quoting rules and break when the code contains `"`, `$`, or `()`. Instead, always write a `.py` file with the Python code as a regular string:

```python
# ✅ CORRECT — Python multi-line string in a .py file
code = r"""
import sys
e = sys.modules["bl_ext.user_default.texturesynth.core.cpp_module"].get_engine()
result = {"status": "ok", "generation": e.current_revision()}
"""

if __name__ == "__main__":
    from mcp_fetch import send
    import json
    resp = send(code)
    print(json.dumps(resp, indent=2))
```

Run it directly:
```powershell
python script.py
```

The `mcp_fetch` module handles the TCP socket and null-byte framing. Your code executes inside **Blender's Python**, not the shell — so shell quoting never matters.

**Never** try to inline multi-line Python via `python -c "@`...`"@` — it breaks on quotes, dollar signs, and nested strings. If you need a one-liner for a quick test, use `mcp_fetch.py` directly:

```powershell
python .opencode/skills/blender-mcp/scripts/mcp_fetch.py "import bpy; result={'ver': bpy.app.version_string}"
```

## Tracing the evaluation timer (dispatch decision chain)

When the addon skips dispatch, read `trace_timer.py` output. The decision at `evaluation.py:175`:

```
needs_dispatch = ready and (_force_render or _params_dirty)
```

**State modules** (all readable from MCP via `sys.modules`):

| Module | File | Readable state |
|---|---|---|
| `...core.evaluation` | `ADDON/core/evaluation.py` | `_topology_dirty`, `_params_dirty`, `_compiling`, `_force_render` |
| `...core.engine_bridge` | `ADDON/core/engine_bridge.py` | `_submitted_generation`, `_last_applied_generation`, `_last_pushed_param_hash`, `_last_active_node_id`, `_last_mute_snapshot`, `_image_cache`, `_pending_image_uploads` |
| `...core.cpp_module` | `ADDON/core/cpp_module.py` | `is_loaded()`, `get_engine()` → `.has_pipeline()`, `.is_generation_ready(gen)`, `.async_in_flight()` |

**Dispatch chain** (which flag sets what):

1. User edits a parameter → `request_param_update()` → `_params_dirty = True`
2. User changes topology → `request_topology_update()` → `_topology_dirty = True` → timer submits graph → sets `_force_render = True`
3. Active node changes → `_on_node_select_change()` or timer poll → sets `_force_render = True` OR calls `request_topology_update()`
4. Timer tick (line 166-183): checks `ready = is_generation_ready(submitted)` → `needs_dispatch = ready and (_force_render or _params_dirty)` → calls `update_params_only(force_submit=True)`
5. **Dead code bug** (`evaluation.py:183-188`): `return 0.016` is before the flag-clearing lines. `_force_render` and `_params_dirty` are **never cleared**. Once set, `needs_dispatch` is permanently True → dispatch fires every 16ms tick.

**Why dispatch might not fire:**
- `engine` is None or `has_pipeline()` is False → `needs_dispatch` never reached
- `_submitted_generation` is 0 → `ready` stays False
- `is_generation_ready(gen)` returns False (compile still in-flight) → `ready = False`
- `_force_render` AND `_params_dirty` both False → nothing triggers a render

## Gotchas

| Gotcha | Mitigation |
|---|---|
| `result` not a dict | Always wrap: `result = {"key": value}` |
| C++ struct not subscriptable | Use `.attr`, not `["attr"]` |
| `bpy_prop_array` | `round(float(val), 6)` before assigning to result |
| `Color` / `Vector` RNA types | `json.dumps(default=repr)` or early convert |
| C++ enum (BitDepth) | `str(e.graph_default_depth())` |
| `texturesynth_core.get_engine()` doesn't exist | `get_engine()` is in `cpp_module`, not at `texturesynth_core` module level. Use `from ...cpp_module import get_engine` |
| One request per connection | Server closes after each response |
| 10s client timeout | Split long loops or use scripts |
| `n.items()` returns empty on TS nodes | Use `n.bl_rna.properties` instead |
| `_` as list-comp iteration var + string literal `'_x'` | Python concatenates: `_[hasattr(e, '_x') for _ in items]` parses as `_x = _` (loop var) + `"x"`. Use `[hasattr(e, '_x') for item in items]` |
| `hasattr()` on C++ binding triggers exception | Nanobind `__getattr__` can raise `RuntimeError` instead of returning False. Use `try: obj.attr; has=True; except (AttributeError, RuntimeError): has=False` |
| MCP server not running | Check Blender → Extensions → MCP is enabled + port 9876 open |
| ImportError after addon enable | Engine `.pyd` not deployed — see AGENTS.md §4 |
