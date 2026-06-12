# AGENTS.md — Mandatory rules for AI coding agents

> **Read before ANY build/clean/install/test or technical explanation.**
> Violating build-cache rules costs internet data. Violating explanation styles makes answers unusable.

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
- *If accidentally wiped*: Note to user and stop!! (Tip: `FETCHCONTENT_FULLY_DISCONNECTED=OFF` repopulates cache if online).

---

## 3. ADDON Architecture: Do Not Duplicate
A complete Blender 4.2+ extension exists at `ADDON/` in the repo root.
**🚫 DO NOT create `src/texturesynth_addon/` or `src/blender_addon/`. Always `ls ADDON/` first.**

---

## 4. Installed Extension Folder is STRICTLY OFF LIMITS
**🚫 DO NOT touch:** `%APPDATA%\Blender Foundation\Blender\{ver}\extensions\user_default\{id}\` without user's permission.
This is the user's working install.
- DO NOT `Copy-Item -Recurse -Force` into it.
- DO NOT edit its `blender_manifest.toml` or delete files.
- DO NOT touch `wheels/` inside it.
**Correct workflow:** Edit files in `ADDON/` (repo root). The user will deploy them manually.

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

---

## 6. Workflow Rules

**🔍 Before Writing Code:**
1. `ls` the directory (know what exists).
2. `grep` for the concept (avoid duplicates).
3. Read target file for style/imports.
4. Check `DEV_LOG/*/` and `AGENTS.md`.
5. Run parallel `explore` subagent if unfamiliar.

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
