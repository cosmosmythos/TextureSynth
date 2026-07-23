"""Resolver helper — paste at the TOP of every other script.

The TextureSynth extension's import path depends on how Blender loaded it:
  - 4.2+ extension:  bl_ext.user_default.texturesynth
  - legacy addon:    texturesynth

Blender MCP's execute_code runs in a fresh {"bpy": bpy} namespace per call,
so imports do NOT persist. Each script must resolve the package itself.

This module is not run standalone — copy the body into the script that needs it.
"""
import sys
import importlib


def _ts_pkg():
    """Return the imported TextureSynth addon package, or raise."""
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        if name in sys.modules:
            return importlib.import_module(name)
    # Walk sys.modules for a package exposing core.cpp_module (already loaded
    # under an unexpected name, e.g. a dev checkout).
    for mod in list(sys.modules.values()):
        core = getattr(mod, "core", None)
        if core is not None and hasattr(core, "cpp_module"):
            return mod
    # Last resort: try the canonical names cold.
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise RuntimeError(
        "TextureSynth addon not loaded. Enable the extension in "
        "Blender > Preferences > Extensions, then retry."
    )


# Self-test when run directly (for sanity, not used by other scripts).
if __name__ == "__main__":
    import bpy  # noqa
    pkg = _ts_pkg()
    core = pkg.core
    print(f"package: {pkg.__name__}")
    print(f"cpp loaded: {core.cpp_module.is_loaded()}")
    print(f"engine: {core.cpp_module.get_engine() is not None}")
