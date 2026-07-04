# ADDON — Blender 4.3+ Extension

## Purpose
The distributable Blender extension (`id = "texturesynth"`, `blender_manifest.toml`). Registers a custom node tree (`TextureSynthTreeType`), generates node classes from the C++ node library, runs an event-driven evaluation loop, and bridges graph state to the `texturesynth_core` engine via async readback.

## Ownership
- `__init__.py` — register/unregister order (root of the addon).
- `blender_manifest.toml` — extension manifest (schema 1.0.0, Blender ≥ 4.3).
- `core/` — `cpp_module` (loads the `.pyd`), `evaluation` (msgbus+timer loop), `engine_bridge` (graph↔engine), `preferences`, `logging`.
- `nodes/` — `tree` (custom node tree), `base`, `factory` (dynamic class generation), `categories`, and `specialized/` (hand-written nodes).
- `nodes/specialized/_common.py` — shared `update_param()` utility used by multiple specialized modules.
- `operators/` — `update`, `connect`, `output_targets`, `bake`.
- `panels/` — `sidebar` UI. Registers Scene-level properties (`texturesynth_resolution`, `texturesynth_precision`).
- `utils/` — `dll_loader` (wheel DLL search), `node_utils`, `rna`.
- `wheels/` — destination for the CI-built wheel (`.gitkeep` only in repo).

## Local Contracts
- **Register order is load-bearing** (`__init__.py:11-27`):
  1. `nodes.tree.register()` (avoids SpaceNodeEditor console noise)
  2. `preferences.register()` (C++ log sink reads the level)
  3. `cpp_module.load()` (import `.pyd`; non-fatal if missing)
  4. `nodes.register()` → `factory.generate_node_classes(cpp_module)` then register UI + `categories.register()` (hooks into `NODE_MT_add`)
  5. `operators.register()`, `panels.register()`
  6. `evaluation.register()` (start timer — must be last, after all node classes exist)
  Unregister is the strict reverse.
- **Never create `src/texturesynth_addon/` or `src/blender_addon/`** (root §3). This `ADDON/` is the only addon source.
- **Installed extension folder**: edit `.py` files in `ADDON/` here; the agent auto-deploys changed `.py` files to the install folder (root §4).
- **🔴 NEVER deploy .pyd/.so binaries.** When C++ changes in `src/` are needed, **ask user for approval first, then commit and push**. Do NOT build locally and copy the `.pyd` anywhere.
- **Stable IDs**: `engine_bridge.py` uses `stable_id()` so graph identity survives Blender node reordering/invalidation. Do not use Blender's unstable pointer/index as the engine `NodeId`.
- **Specialized node contract** (`nodes/specialized/__init__.py`): each module may export `SV_TYPE`, `NODE_CLASS` (or `NODE_CLASSES` tuple), `SOCKET_CLASSES`, `PROPERTY_GROUPS`, `OPERATOR_CLASSES`. PropertyGroups MUST register before classes that reference them (root §5 gotcha). The factory skips any `sv_type` in `specialized_sv_types()`.
- **Mute rewiring** (root §5 gotcha): `resolve_muted_source()` must search ALL inputs, not just `input[0]`, because control inputs (mask/gain) sit at socket 0.
- **Engine lifecycle**: `cpp_module.shutdown()` before re-`load()`. One VkInstance per process.
- **Image upload caching** (`engine_bridge.py`): `upload_node_image()` compares a cheap signature (`as_pointer, w, h, source, is_dirty`) against `_image_cache` before calling `foreach_get`. Content hash fallback only runs on `is_dirty` flip (edit saved). Cache cleared on `unregister()`. Ring-full retries queue in `_pending_image_uploads` and drain on next `submit_graph()`.

## Work Guidance
- After edits: clean Python cache (root §6) — only under `ADDON/`.
- If `ADDON/shader_assets/` exists, delete it — it's a runtime artifact, never checked in. Same rule as `__pycache__`.
- Blender MCP can run Python in a live Blender for testing node behavior — useful for graph-building/debug work.
- New node with custom UI: add a module under `nodes/specialized/`, append its name to `_MODULE_NAMES`, export the required symbols.
- Dynamic (non-specialized) nodes are generated from the C++ `NodeLibrary` via `engine.node_library().all()` when the engine is available. Falls back to reading `shader_assets/nodes/*.node.json` when the engine is unavailable.

## Verification
- `tests/python/` — pytest against `texturesynth_core` (skips if Vulkan init fails): `test_bake.py`, `test_graph.py`, `test_images.py`, `test_lifecycle.py`, `test_node_library.py`, `test_params.py`, `test_render.py`.
- Run: `pytest tests/python/` (or `run_tests.bat`); asset paths resolved via `conftest.py`.

## Child DOX Index
None yet — `core/`, `nodes/`, `operators/`, `panels/` are cohesive enough to share this doc. If `nodes/specialized/` grows complex (many hand-written nodes with shared patterns), split a child doc there.
