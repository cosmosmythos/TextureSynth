# AGENTS.md — Mandatory rules for AI coding agents

> **Read this file before running ANY build/clean/install/test command.**
> Violating these rules will cost the user real internet data and several
> minutes of redownload time. Every coding agent makes the same mistake
> once; this file exists to make sure you don't.

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
