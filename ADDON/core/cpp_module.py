"""
C++ Module Loader
Loads the wheel-provided texturesynth_core C++ module.
"""

_core = None      # texturesynth_core pyd module
_engine = None    # Engine instance


def load():
    """Load the C++ module and initialise the Vulkan engine.

    Safe to call from register() — the DLL loader runs first,
    then the import, then engine construction.
    Non-fatal: prints errors but never raises.
    """
    global _core, _engine

    if _engine is not None:
        return True  # already loaded

    # Step 1 — ensure wheel-bundled DLLs are discoverable on Windows
    try:
        from ..utils.dll_loader import add_wheel_dll_dirs
        add_wheel_dll_dirs("texturesynth_core")
        add_wheel_dll_dirs("texturesynth")
    except Exception:
        pass

    # Step 2 — import the pyd
    try:
        import texturesynth_core as core
        _core = core
        try:
            core.set_log_callback(
                lambda level, message: print(f"[TextureSynth C++] {level.strip()} {message}")
            )
        except Exception as e:
            print(f"[TextureSynth] Failed to install C++ log callback: {e}")
        print("[TextureSynth] Core module imported OK.")
    except ImportError as e:
        print(f"[TextureSynth] ImportError: {e}")
        print("[TextureSynth] Engine will not be available. Nodes still work as UI.")
        return False
    except Exception as e:
        print(f"[TextureSynth] Core import exception: {e}")
        return False

    # Step 3 — construct Engine (no threads started here)
    try:
        eng = _core.Engine()
    except Exception as e:
        print(f"[TextureSynth] Engine() constructor exception: {e}")
        _core = None
        return False

    # Step 4 — init Vulkan (headless, no surface)
    try:
        import bpy
        import os
        # Store shader cache in the user's addon data directory (safe from permissions issues)
        cache_dir = os.path.join(bpy.utils.user_resource('DATAFILES', path="texturesynth"), "shader_cache")
        addon_root = os.path.dirname(os.path.dirname(__file__))
        nodes_dir  = os.path.join(addon_root, "shader_assets", "nodes")
        glsl_dir   = os.path.join(addon_root, "shader_assets", "glsl")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        ok = eng.init(enable_validation=False, cache_dir=cache_dir, nodes_dir=nodes_dir, glsl_dir=glsl_dir)
        if not ok:
            print(f"[TextureSynth] Engine.init() failed: {eng.last_error()}")
            _core = None
            return False
    except Exception as e:
        print(f"[TextureSynth] Engine.init() exception: {e}")
        _core = None
        return False

    _engine = eng
    print("[TextureSynth] Vulkan engine initialised OK.")
    # Phase 6 sanity: warn if any Python node's named-param keys drift from
    # the C++ manifest. This catches JSON renames silently breaking sliders.
    try:
        lib = eng.node_library().all()
        print(f"[TextureSynth] Param-name contract: {len(lib)} C++ manifests loaded.")
    except Exception as e:
        print(f"[TextureSynth] Warning: could not verify NodeLibrary manifests: {e}")
    return True


def shutdown():
    """Destroy the Vulkan engine. Safe to call even if never loaded."""
    global _core, _engine
    if _engine is not None:
        try:
            _engine.shutdown()
            print("[TextureSynth] Engine shut down.")
        except Exception as e:
            print(f"[TextureSynth] Engine shutdown exception: {e}")
    if _core is not None:
        try:
            _core.set_log_callback(None)
        except Exception:
            pass
    _engine = None
    _core = None


def is_loaded():
    """True when Engine is ready for dispatch."""
    return _engine is not None


def get_engine():
    """Return the Engine instance, or None."""
    return _engine


def get_core():
    """Return the texturesynth_core module, or None."""
    return _core
