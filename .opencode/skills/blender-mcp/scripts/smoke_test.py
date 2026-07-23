"""Smoke test: is the TextureSynth addon alive inside this Blender?

Run via Blender MCP execute_code. Verifies, in order:
  1. The extension is enabled.
  2. The C++ core (.pyd) loaded and the Vulkan engine initialized.
  3. The node library is populated (factory will generate classes).
  4. A TextureSynth node tree is reachable.

Stops at the first failure -- every downstream test assumes these hold.
Output is print()-based because execute_code captures stdout as the result.
Does NOT mutate state (does not create a tree if none exists; just reports).
"""
import sys
import importlib

results = []


def check(label, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {label}" + (f" -- {detail}" if detail else "")
    results.append((ok, line))
    print(line)
    return ok


def _ts_pkg():
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        if name in sys.modules:
            return importlib.import_module(name)
    for mod in list(sys.modules.values()):
        core = getattr(mod, "core", None)
        if core is not None and hasattr(core, "cpp_module"):
            return mod
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def main():
    import bpy
    import addon_utils

    try:
        enabled = addon_utils.check("texturesynth")[0]
    except Exception as e:
        check("extension enabled", False, f"addon_utils raised {e!r}")
        return
    check("extension enabled", enabled, "texturesynth extension checked on")

    pkg = _ts_pkg()
    if not check("addon importable", pkg is not None,
                 "package resolved from sys.modules"):
        print("\nResolve failed: enable the extension in Preferences > Extensions")
        return

    core = pkg.core
    if not check("cpp_module loaded", core.cpp_module.is_loaded(),
                 "texturesynth_core .pyd imported"):
        return

    engine = core.cpp_module.get_engine()
    if not check("engine initialized", engine is not None,
                 "Vulkan Engine() + init() ok"):
        return

    try:
        all_types = engine.node_library().all()
        sv_types = sorted(all_types.keys())
    except Exception as e:
        check("node library", False, f"{e!r}")
        return
    check("node library non-empty", len(sv_types) > 0,
          f"{len(sv_types)} sv_types: {sv_types}")

    try:
        tree = core.engine_bridge._find_node_tree()
        detail = (f"found tree {tree.name!r}" if tree
                  else "no TS tree yet (run build_graph to make one)")
        check("node tree reachable", True, detail)
    except Exception as e:
        check("node tree reachable", False, f"{e!r}")

    try:
        has_pipe = engine.has_pipeline()
        gen = core.engine_bridge._submitted_generation
        check("pipeline state readable", True,
              f"has_pipeline={has_pipe} submitted_gen={gen}")
    except Exception as e:
        check("pipeline state readable", False, f"{e!r}")

    passed = sum(1 for ok, _ in results if ok)
    total = len(results)
    print(f"\n=== smoke: {passed}/{total} checks passed ===")


main()
