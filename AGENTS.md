# AGENTS.md — Mandatory rules for AI coding agents

**Read before ANY build/clean/install/test, or before writing a technical explanation.**
Skipping the cache rules wastes bandwidth and rebuild time. Skipping the explanation rules makes answers unusable.

> **🔴 CRITICAL: NEVER DEPLOY .pyd/.so BINARIES**
> Do NOT copy, build, or place `.pyd`/`.so` files into the Blender install folder, `wheels/`, or anywhere else for testing.
> GitHub CI builds and zips the addon. Edit source files in `ADDON/` and `src/`, **ask user for approval before committing**, then commit and push — let CI produce the distributable.
> Local `.pyd` copies bypass CI validation and cause silent breakage.
> If you build a `.pyd` locally, **leave it in `build/Release/`** and ask user for approval to commit+push.

## Skill Usage

Invoke relevant or requested skills BEFORE any response or action. Even a 1% chance a skill might apply means that you should invoke the skill to check. If an invoked skill turns out to be wrong for the situation, you don't need to use it.

## 0. STRICT BEHAVIOR: Root-cause or silence
When something is broken:

1. **Trace the full chain** — from the root definition, files, through every intermediate layer, to the final consumer. Check each layer, don't skip ahead.
2. **Read the root definition first** (the schema, type, interface, or config everything else derives from) BEFORE touching the code that consumes it.
3. **Fix every instance of a pattern in one pass.** If a value or check is wrong in several places, investigate deeply & fix all of them together. Never "fix one and see if it works."
4. **If a patch "works" but you haven't verified the root cause is correct, STOP.** The underlying design may be broken — a symptom fix just hides the real bug.
5. **Never assume — search the whole codebase** for every spelling/variant of the broken pattern (different naming conventions, prefixes, abbreviations) before concluding you've found them all.
6. **When adding a new attribute or function**, audit every existing file and call site that creates/touches the same data. Don't add it and hope.
This is non-negotiable.

**Debugging rule: When a fix "has no effect", stop guessing and output actual values.**
Trace backwards from the broken output, injecting known values at each layer until you find where reality diverges from expectation.

## 1. Technical Explanations & Style
The user is a **3D texture artist/developer**. Every architecture/design answer MUST include these 3 layers in order:

1. **Workflow/Pipeline** (4-12 lines)
   - Use TEXTURESYNTH terms: *graph, node, chain, output_node, dispatch, image, preview, slider*
   - Use Blender terms: *node editor, readback*
   - Include a small table.
   - Vulkan, c++ etc. code explanations(simple, like to a person learning programming)
2. **Architecture / Data Flow**
   - Which structs/files change. Cite `file:line` & it's connections to the rest of texturesynth code and why.
3. **Code/Pseudo-code Example**
   - GLSL/C++/JSON snippets. Walk through a concrete example.

**Hard Style Rules:**
- NO filler ("let me explain", "in this response"). Use tables, bullets, short paragraphs.
- Always ground abstract terms: e.g., "thread-local register = a local variable in the shader = a slot on the GPU chip that holds 4 floats".
- If user says "I still don't get it", "I can't picture this" etc., use simple flowchart/diagrams.
- **Plan mode**: Use this style for analysis (do not edit files).
- **Build mode**: Apply style AND make file changes without surrounding commentary.
- **No third-party software names in code.** Source files (`.py`, `.glsl`, `.json`, `.cpp`, `.hpp`) must NOT reference Nuke, Substance Designer, Photoshop, Houdini, Krita, Unity, Unreal, etc. Describe the *math* (formula, inputs, behavior), not which app inspired it. Conversations and DOX docs may reference them for context; code and code comments may not.
- **Code is not a tutorial.** The codebase is for developers who already understand the objective. Do NOT write long docstrings, "what this does" explanations, or block comments restating behavior. Prefer `None` or empty tooltips. Keep comments to a line, or none — let clear naming carry it.
- **glslViewer_tests is a frozen mirror.** Do not hand-edit copies there while actively developing a node. Edit the canonical source in `shader_assets/glsl/` (and `shader_assets/nodes/`); mirror to `glslViewer_tests/` once, at the end, when the node is final.

---

## 2. Build Commands & The Cache Trap
**`build/_deps/` is a sacred FetchContent cache.** Re-fetching burns mobile data.

**🚫 NEVER run these (wipes cache):**
- `rmdir /s /q build`
- `rm -rf build`
- `cmake --build . --target clean` (wipes `_deps` artifacts)
- `./build_clean.bat` (The trap script)
- `git clean -fdx build/`
- `Remove-Item -Recurse build`

**✅ ALWAYS use these instead:**
- `./build_fast.bat` (**DEFAULT** Incremental rebuild)
- `cmake --build build --config Release --target <name>` (Rebuilds ONE target)
- `cmake -S . -B build` (Refresh settings safely)

**If you get CMake errors (missing target/source/dep), DO NOT WIPE. Instead:**
1. Re-detect generator: `cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_PYTHON_BINDINGS=ON`
2. Re-configure bad var: `cmake -S . -B build -DBUILD_TESTS:BOOL=ON`
3. Rebuild specific target: `cmake --build build --config Release --target texturesynth_core`

**When to wipe `build/` (WARN USER FIRST):**
- User explicitly says "nuke build"
- MSVC compiler upgrade or FetchContent `GIT_TAG` change
- *If accidentally wiped*: Note to user and stop!! (Tip: `FETCHCONTENT_FULLY_DISCONNECTED=OFF` repopulates cache if online)

---

## 3. ADDON Architecture: Do Not Duplicate
A complete Blender 4.2+ extension exists at `ADDON/` in the repo root.
**🚫 DO NOT create `src/texturesynth_addon/` or `src/blender_addon/`. Always `ls ADDON/` first.**

---

## 4. Installed Extension Folder: Standing Deploy Exception
**Install root:** `%APPDATA%\Blender Foundation\Blender\5.0\extensions\user_default\texturesynth\`

After editing `.py` files in `ADDON/`, **copy only the changed `.py` files** to the install folder automatically so the user can manually test in Blender fast. No need to ask each time.

### What auto-deploys (overwrite install copy from repo source)

| Repo source (edit here) | Install destination (copy here) |
|---|---|
| `ADDON/**/*.py` | `<install>/**/*.py` (mirror path under install root) |

**Only `.py` files.** Do NOT deploy `.pyd`, `.so`, `.glsl`, `.node.json`, `.toml`, or any other file type.

Deploy only the files you changed — no blanket `Copy-Item -Recurse -Force` of whole folders.

### Still off-limits (NEVER touch in the install folder)

- `blender_manifest.toml` — edit in `ADDON/` only.
- `wheels/` and any `.pyd` / `.so` binaries — **NEVER copy a locally-built .pyd to the install folder.** GitHub CI builds and zips the addon. Edit source files in `ADDON/` and `src/`, let CI produce the distributable. Local .pyd copies bypass CI validation and cause silent breakage. **When C++ changes are needed: ask user for approval first, then commit and push.**
- `core/` C++ binding sources — rebuild via `build_fast.bat`, do not hand-edit.
- `shader_assets/` — GLSL and JSON node manifests live in the repo; the engine loads them at init. Do not copy them to the install folder.

### Trigger

Run the deploy when a `.py` edit to `ADDON/` is made. State what was copied in the reply.

**Implementation:** on Windows, use PowerShell:
```powershell
Copy-Item "ADDON\<path>.py" "$env:APPDATA\Blender Foundation\Blender\5.0\extensions\user_default\texturesynth\<path>.py" -Force
```
On bash-capable systems, the `blender-addon` OpenCode skill (`.opencode/skills/blender-addon/`) owns the deploy script. Section 4 and the skill are kept in sync — do not weaken one without updating the other.

---


---

## 5. Language / API Gotchas
| Gotcha | Mitigation |
|---|---|
| `EnumXxx.None` won't parse in Python | Use `EnumXxx['None']` (Python keyword issue). |
| `\u2014` (em-dash) in C++ strings | Use ASCII `--` (prevents MSVC cp1252 module import crash). |
| `nb::ndarray` vector backing | Use `nb::capsule` owner with `new[]`/`delete[]` to prevent dangling pointers. |
| Multiple `Engine()` instances crash | One VkInstance per process. Call `shutdown()` before new engines. |
| PushConstant seed type | Python's `tc.PushConstants.seed` is `uint32_t`. Pass `int`, not `float`. |
| C++ `engine_tests` OFF by default | Reconfigure with `-DBUILD_TESTS:BOOL=ON`. |
| `IntProperty` truncates `uint64_t` | Store ID as `StringProperty` hex, convert at use site. |
| `CollectionProperty(type=PropertyGroup)` | Register target `PropertyGroup` BEFORE dependent node classes. |
| Muted node rewiring socket index | `resolve_muted_source()` must search ALL inputs, not just input[0], to find data connections (control inputs like mask/gain are at socket 0). |
| FusedVariantKey stale cache | Cache key MUST include every GLSL-affecting field. Three collision classes each shipped a bug: (1) `external_inputs` count alone was insufficient — two graphs with same count but different socket mappings shared the same key; fix = per-node `external_socket_masks`. (2) same node types with swapped internal A/B wiring (e.g. `Blend(Value→a, Simplex→b)` vs `Blend(Simplex→a, Value→b)`) emit different GLSL but collided on every existing field; fix = `internal_producer_indices` (flat per-socket producer local_index, `UINT32_MAX` for non-RegSrc). (3) residual: unconnected-socket `ConstSrc{value}` defaults baked into GLSL but not in key — flagged for future work. When adding a field, update `FusedVariantKey` (decl + `==` + `hash()`), `build_fused_key()`, the emitter→`Chain`→key propagation path, `ShaderCache::write_sidecar_()`/`sidecar_matches_()`, then bump `epoch`. Current fields: `node_type_ids`, `param_socket_masks`, `input_counts`, `feature_flags` (3 bits format + 2 bits depth per node), `external_socket_masks`, `internal_producer_indices`, `epoch=8`. ShaderVariantKey epoch=5 (3 bits format + 2 bits depth in feature_flags). |
| Fusion cross-group producer missing image | When a chain splits, intermediate (non-tail) nodes feed in-chain consumers via registers but **cross-chain consumers via VRAM textures**. `FusedGraphCompiler` MUST scan every pass's input_resources and insert any cross-group producer's output into `active_resources`, or `ResourceManager` skips its image and the downstream chain reads garbage. The planner's DAG MUST use real `ir.connections` edges (not synthetic linear `path[i-1]->path[i]`), and `is_valid_path` must accept any topological order (fan-out OK). |
| Storage format = channels x depth | `ChannelFormat` (Mono/UV/RGB/RGBA/ID/Metadata) and `BitDepth` (F8/F16/F32) are orthogonal axes composed via `StorageFormat{channels, depth}`. The GLSL layout qualifier and VkImage allocation MUST both derive from the same `StorageFormat` via `storage_format_glsl_qualifier()` and `storage_format_to_vk()` -- never hardcode format literals. GLSL uses `f` suffix (`rgba16f`), NOT VkFormat's `_SFLOAT` spelling. Depth resolves via SD-style inheritance (`resolve_node_depths`): Auto -> graph default from sidebar, MatchInput -> upstream's resolved depth, Absolute -> node's `absolute_depth`. Single dummy image (1x1 RGBA32F) serves all formats -- texelFetch on sampled image views does automatic format conversion per Vulkan spec. |

---

## 6. Workflow Rules

**🚫 NEVER deploy .pyd/.so binaries manually.** GitHub CI builds and zips the addon. Edit source files in `ADDON/` and `src/`, let CI produce the distributable. **When C++ changes are needed: ask user for approval first, then commit and push.**

**🔍 Before Writing Code:**
1. `ls` the directory (know what exists).
2. `grep` for the concept (avoid duplicates). **Always search `src/` and `ADDON/` and `shader_assets/` separately** — a term not in the repo root doesn't mean it's absent from engine code.
3. Read target file for style/imports.
4. Check `DEV_LOG/*/` and `AGENTS.md`.
5. Run parallel `explore` subagent if unfamiliar.

**🎨 Blender Integration (MCP):**
- Use Blender MCP to run Python scripts/tests directly in Blender environment
- Useful for testing addon behavior, debugging graph building, and verifying node behavior
- Run scripts with live Blender context: `import bpy`, access `bpy.context.scene`, etc.

**🧹 Clean Python Cache (Run after tests/before handoff):**
```powershell
Get-ChildItem -Path .\ADDON -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
```
*(Do not clean `__pycache__` outside `ADDON/` like `tests/python/`, `build/`, or `shader_assets/`)*

---

## 7. Code Style: No Verbose Commentary
- **No huge blocks of comments** in code. Code should be self-documenting via clear naming.
- **No tutorial-style docstrings** that re-state what a function does. One-line docstrings are fine.
- **3-layer explanations** (artist/architecture/code) are for *conversations with the user*, not for source files.
- When the user says "no commentary in code" — they mean it literally. A single sentence in a docstring is OK; a paragraph is not.
- `git diff` should show *behavior changes*, not comment changes. If your diff is mostly comment edits, delete them.
- **Inline comments are max 1 line.** If you need more, rename the symbol or extract a function — don't write a paragraph.

---

# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Update child docs when parent changes alter local rules
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete SURGICAL details in child docs
- DO NOT duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist
- The closer a doc is to the work, the more specific and practical it must be

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
- Update parent docs when parent-level structure, ownership, workflow, or child index changes
- Update child docs when parent changes alter local rules
- Remove stale or contradictory text
- Run existing verification when relevant
- Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Child DOX Index

TextureSynth is a node-based Vulkan procedural-texture engine with a Python/Blender frontend. The repo splits cleanly into four durable boundaries: C++ engine source, the Blender addon, shader assets, and tests. Each child doc owns its subtree; the root owns project-wide rules (build/cache, style, gotchas).

| Child doc | Owns | Key invariant |
|---|---|---|
| [`src/AGENTS.md`](src/AGENTS.md) | C++20 source: static `engine` lib, `texturesynth_core` nanobind extension, optional `viewer` exe | One `VkInstance` per process; `engine` lib must not depend on bindings/viewer |
| [`src/engine/AGENTS.md`](src/engine/AGENTS.md) | Vulkan compute engine core (Graph→IR→PassPlan→dispatch→readback) | Pipeline stages flow strictly Graph → GraphIR → PassPlan → PassExec → dispatch |
| [`src/engine/graphfusion/AGENTS.md`](src/engine/graphfusion/AGENTS.md) | Chain fusion: DAG→Planner→Emitter→FusedGraphCompiler | Fused path must be bit-identical to the unfused reference per-node |
| [`src/engine/register_allocation/AGENTS.md`](src/engine/register_allocation/AGENTS.md) | Graph-coloring register allocator for fused shader chains | Interval graphs are perfect; linear scan is optimal for our DAG liveness |
| [`ADDON/AGENTS.md`](ADDON/AGENTS.md) | Blender 4.3+ extension (`ADDON/`): register order, nodes/operators/panels, engine bridge | `cpp_module` loads `.pyd` from wheel; never create `src/*_addon/` |
| [`shader_assets/AGENTS.md`](shader_assets/AGENTS.md) | Node manifests (`*.node.json`), GLSL node fns, common GLSL | Every node GLSL follows the `vec4 node_<name>(vec2 uv, ...)` signature contract |
| [`tests/AGENTS.md`](tests/AGENTS.md) | C++ gtest suite + Python pytest suite against the binding | C++ tests need `-DBUILD_TESTS:BOOL=ON`; Python tests skip if Vulkan init fails |
| [`DEV_LOG/AGENTS.md`](DEV_LOG/AGENTS.md) | User's dev journal: roadmaps, feature plans, architecture notes | Read-only for agents; no verification |

Hierarchy:
```
AGENTS.md  (this file — global rules: cache, style, gotchas, DOX)
├── src/AGENTS.md
│   └── src/engine/AGENTS.md
│       ├── src/engine/graphfusion/AGENTS.md
│       └── src/engine/register_allocation/AGENTS.md
├── ADDON/AGENTS.md
├── shader_assets/AGENTS.md
├── tests/AGENTS.md
└── DEV_LOG/AGENTS.md
```

Cross-cutting artifacts that live at repo root (no dedicated doc — covered by root rules):
- `CMakeLists.txt`, `cmake/Dependencies.cmake` — build graph; see §2 (cache trap).
- `build_fast.bat` (use) / `build_clean.bat` (never use) — see §2.
- `.github/workflows/build_texturesynth.yml` — CI builds wheels for py311/py313.
- `deploy.ps1`, `scripts/*.bat` — local helper scripts; non-durable.
