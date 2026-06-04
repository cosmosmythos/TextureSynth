# AGENTS.md — Mandatory rules for AI coding agents

> **Read this file before running ANY build/clean/install/test command, AND
> before answering any non-trivial technical question.**
> Violating the build-cache rules costs real internet data. Violating the
> explanation-style rules makes the answers unusable for the project owner.

---

## How to explain technical work (read this before answering design questions)

The project owner is a **3D texture artist who is also a developer and the
architect of this tool**. They are building TEXTURESYNTH to release as a
**state-of-the-art industry tool inside Blender** — scalable, robust, fast,
artist-friendly, non-blocking. Every explanation must serve that goal.

When answering design / architecture / "how does X work" questions, every
answer MUST include **three layers** in this order, in the same message:

1. **Artist mental model first** (always)
   - Use TEXTURESYNTH terms: graph, node, chain, output_node, dispatch, image, preview, slider
   - Use Blender terms an artist knows: node editor, materials, render, viewport
   - No analogies from outside domains (no "coffee shop", "car wash", "factory assembly line" — the project owner has explicitly rejected these)
   - Show what the artist *sees* and *does* — clicks, drags, previews, errors
   - "What changes for me as the artist?" answer in a small table

2. **Architecture / data flow** (always)
   - Which structs change, which files change, which engine stage is affected
   - The artist sees the *result* of this layer; they don't see the layer itself
   - Cite `file:line` for the code that implements it

3. **Actual code or pseudo-code** (always, for non-trivial work)
   - GLSL snippets for shader work
   - C++ snippets for engine work
   - JSON / manifest snippets for data
   - One concrete example (perlin → invert → grayscale, or similar) walked through end to end

**Hard rules for the explanation style**:
- No filler. No "let me explain". No "in this response I will". Start with the table or the answer.
- Use tables, bullets, short paragraphs. No walls of prose.
- Length: 4-12 short lines for the artist layer, then the technical layers can be longer.
- When using a phrase like "thread-local register", always follow it with "= a local variable in the shader = a slot on the GPU chip that holds 4 floats" — connect the abstract to the concrete the artist already knows from C++ / Blender.
- If a question is conceptual (e.g., "what does this mean for me?"), the artist layer is the WHOLE answer. Don't add a "but here's the technical detail" footnote unless asked.
- When the user asks "what changes for the artist day-to-day?", answer that FIRST and ONLY. Don't drift into GPU bandwidth numbers unless the user asks.
- If the user says "I can't picture this" or "doesn't make sense" or "break it down" — they mean the artist layer is missing, not that the technical layer is wrong. Reset to the artist layer and rebuild from there.

**When the user is in plan mode** (read-only), use this style for analysis,
recommendations, and option comparisons. Do NOT make file edits.

**When the user is in build mode**, apply this style and ALSO make the
file changes / run the build / write the code the user asked for. Do not
add commentary before/after the work — let the changes speak.

---

## 🚫 NEVER run these commands

---

## 🚫 NEVER run these commands

| Don't run | Why |
|---|---|
| `rmdir /s /q build` | Deletes `build/_deps/` cache (5 deps: json, vk-bootstrap, vma, imgui, glfw) |
| `rm -rf build` | Same as above |
| `cmake --build . --target clean` (at root) | Wipes `_deps` artifacts |
| `./build_clean.bat` | The trap script — see below |
| `git clean -fdx build/` | Same as `rm -rf build` |
| `Remove-Item -Recurse build` (PowerShell) | Same |

## ✅ Always use these instead

| Use | What it does |
|---|---|
| `./build_fast.bat` | **Incremental** rebuild — reuses `build/_deps/`, only recompiles what changed. THIS IS THE DEFAULT. |
| `cmake --build build --config Release --target <name>` | Rebuilds ONE target; deps cache untouched. |
| `cmake -S . -B build` (re-configure) | Refreshes CMake settings without touching `_deps/`. |

The `build/_deps/` directory is a **FetchContent cache** populated once.
Re-downloading it takes ~30-90 seconds and burns mobile data on this machine.
**Treat `build/_deps/` as sacred.**

---

## Why the trap exists

There are two build scripts in the repo root:

- `build_fast.bat` — **safe**. Incremental. Default. Use this.
- `build_clean.bat` — **dangerous**. Wipes `build/` including `_deps/`. The
  name is misleading — "clean" sounds safe but it costs a full re-fetch.

If a CMake error mentions a missing target, missing source, or stale
dependency, the answer is **never** to wipe `build/`. The answer is one of:

1. `cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DBUILD_PYTHON_BINDINGS=ON` to re-detect generator
2. Re-configure with the specific cache var that's wrong (e.g. `cmake -S . -B build -DBUILD_TESTS:BOOL=ON`)
3. Rebuild only the affected target: `cmake --build build --config Release --target texturesynth_core`

---

## When you DO need to wipe `build/`

Only when:

- The user explicitly says "clean everything" or "nuke build"
- Compiler version changes (e.g. upgrading MSVC)
- FetchContent `GIT_TAG` changes for a dependency

In that case, **warn the user first** that this costs a full re-fetch and
proceed only with their explicit OK.

---

## What to do if you accidentally wiped `build/`

Stop. Apologize to the user. Don't try to silently re-download — let them
decide whether to wait on the connection. Future-proofing: when configuring
a fresh build, set `FETCHCONTENT_FULLY_DISCONNECTED=OFF` only if offline mode
is needed; otherwise the first configure will repopulate `_deps/`.

---

## Other agent gotchas in this repo

| Gotcha | Mitigation |
|---|---|
| `EnumXxx.None` won't parse in Python | Use `EnumXxx['None']` (string indexer) — `None` is a Python keyword |
| `\u2014` (em-dash) in narrow string literals → MSVC converts to cp1252 → crashes module import | Use ASCII `--` in docstrings/messages |
| `nb::ndarray<nb::numpy, float, nb::shape<-1,-1,4>>` returns data that dangles if backed by a stack vector | Use `nb::capsule` owner with `new[]`/`delete[]` |
| Multiple `Engine()` instances crash on Windows (only one VkInstance per process) | Tests use a session-scoped `engine` fixture; create new engines only after `shutdown()` |
| Python's `tc.PushConstants.seed` is `uint32_t` | Pass `int`, not `float` |
| C++ test target `engine_tests` is OFF by default | Reconfigure with `-DBUILD_TESTS:BOOL=ON` (the second time, since first cache-write may be ignored) |
| `bpy.props.IntProperty` is signed int32 and silently truncates `uint64_t` `NodeId` | Store as `StringProperty` hex, convert at use site |
| Blender `Node.mute` must map to engine `muted` (rewire / pass-through), NOT `bypassed` (output=zero) | `ADDON/core/engine_bridge.py:202-203` — `muted=bool(mute), bypassed=False` |

---

## 🚫 DO NOT create parallel addon structure

**Lesson learned**: this repo already has a complete Blender 4.2+ extension
at `ADDON/` in the **repo root** (NOT in `src/`). It has its own
`blender_manifest.toml`, `__init__.py`, `core/`, `nodes/`, `operators/`,
`panels/`, `utils/`, `wheels/`. **Always `ls ADDON/` first and read
the existing files before writing any addon-side code.**

Things that already exist and must be REUSED, not rewritten:
- `ADDON/core/cpp_module.py` — singleton engine loader (DLL paths)
- `ADDON/core/engine_bridge.py` — the ONLY file that talks to `texturesynth_core`
- `ADDON/utils/dll_loader.py` — Windows DLL search-path setup
- `ADDON/operators/update.py` + `connect.py` — the standard operators
- `ADDON/nodes/base.py::TextureSynthNode.stable_id()` — uint64 ID derivation
- `ADDON/blender_manifest.toml` — declared at repo root

**If you think you need a new folder like `src/texturesynth_addon/` or
`src/blender_addon/` — STOP. You are duplicating `ADDON/`.**

---

## 🚫 DO NOT add `wheels = [...]` to source `ADDON/blender_manifest.toml`

**Lesson learned**: the GitHub workflow `.github/workflows/build_texturesynth.yml`
**dynamically appends** `wheels = [...]` to the manifest at packaging
time (line 215-216). The source manifest must stay clean so the CI step
can rewrite it. Hardcoding local paths in the source manifest breaks
releases and pollutes the file with OS/ABI-specific entries.

The correct workflow:
1. CI builds the wheel on `windows-latest` for `cp311-cp311` + `cp313-cp313`
2. CI copies `ADDON/` to a build dir, copies the wheels into `wheels/`,
   appends `wheels = [...]` to the manifest
3. CI zips the result into `texturesynth-{version}-windows-x64.zip`
4. CI creates a GitHub release with the zip

**For local testing**: copy the source `ADDON/` to
`%APPDATA%\Blender Foundation\Blender\{ver}\extensions\user_default\texturesynth\`
and install the wheel manually into Blender's bundled Python
(`<blender>\4.5\python\bin\python.exe -m pip install dist\texturesynth-*.whl`).
**Note**: install to Blender's site-packages requires write permission to
`C:\Program Files\...` (or `--user` to user site-packages if Blender's
`sys.path` includes it). **If the local install path is blocked** (DLL
load failures, permission errors), do NOT keep fighting it — push to
GitHub and let CI build the release zip.

---

## 🚫 DO NOT add PNG/file writers to the engine

**Lesson learned**: the engine is a **graph executor**. It produces raw
float pixels. **The host (Blender / viewer / CLI) owns the file format
and the file I/O.** Do not link `stb_image_write`, `libpng`, or any
other image writer into the engine binary.

Why:
- Industry standard: engines return data, hosts persist data. See
  Cycles, Eevee, Substance — none of them write PNGs from the renderer.
- Cross-language: the binding returns an `ndarray` (numpy) and the host
  writes whatever format the user asked for.
- Testable: an engine that returns arrays can be unit-tested with
  pixel-equality assertions. An engine that writes files is impossible
  to test cleanly.

Concretely: `Engine::bake()` returns `std::vector<BakedImage>` where each
`BakedImage { name, width, height, std::vector<float> pixels }` (RGBA32F,
row-major). The Blender `Bake` operator reads these arrays and writes
them to `bpy.data.images` (`float_buffer=True`).

---

## 🔍 Before writing code: investigate the existing code deeply

**Lesson learned (recurring)**: agents tend to start writing new code
before understanding what's already there. This has cost real time on
this project (e.g. creating `src/texturesynth_addon/` when `ADDON/`
existed at repo root).

Mandatory first steps before any non-trivial edit:

1. **`ls` the relevant directory** — know what already exists
2. **`grep`** for the symbol/concept you're about to add — see if it's
   already implemented under a different name
3. **Read the file you're about to edit** — understand the surrounding
   imports, the naming conventions, the style
4. **Check `DEV_LOG/IMPLEMENTATION_PLAN_*/` and `AGENTS.md`** — there's
   almost always a note about the area you're touching
5. **Run a parallel `explore` subagent** for unfamiliar code areas —
   they're cheap and prevent blind edits

If you find that what you wanted to add is already there, USE IT, don't
re-implement it. If it's close but wrong, FIX IT in place. New folders
are a code smell — the existing structure was designed intentionally.

---

## Engine architecture invariants (don't break these)

| Invariant | Why |
|---|---|
| Engine produces **raw float pixels**, no file I/O | Industry standard, testable, host-owned format |
| `bake()` returns `vector<BakedImage>`, not paths | Bindings convert to numpy / bpy.data.images |
| Blender `Node.mute` → engine `muted` (NOT `bypassed`) | `muted` = rewire upstream (pass-through); `bypassed` = output zero (different concept) |
| `output_node` is the **active/preview source** | The `bpy.context.active_node` (or `Output` node's `Result` input) |
| `output_targets[]` are **named bake targets** | PBR Render pattern: Base Color / Normal / Roughness / ... |
| `ShaderCache` sidecar = `<hash>.spv` + `<hash>.spv.key.json` | Cache invalidation requires the FULL FusedVariantKey in the sidecar |
| `Graph::OutputTarget { NodeId source_node, std::string name }` | Stored in `Graph` so it ships in the submit payload |
| `Engine::BakedImage { name, w, h, std::vector<float> pixels }` | RGBA32F, row-major, no padding |
