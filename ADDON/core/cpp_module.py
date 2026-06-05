"""C++ Module Loader: loads texturesynth_core C++ module and initializes engine."""
import bpy
import os

from . import logging as _tslog

_core = None      # texturesynth_core pyd module
_engine = None    # Engine instance


def _cpp_log_sink(level, message):
    """Log filter callback filtering C++ logs by addon preferences."""
    lvl = level.strip().strip("[]").strip().upper()
    if lvl == "INFO" and not _tslog.is_enabled_for("INFO"):
        return
    if lvl == "WARN" and not _tslog.is_enabled_for("WARNING"):
        return
    if lvl == "ERROR" and not _tslog.is_enabled_for("ERROR"):
        return
    print(f"[TextureSynth C++] {level.strip()} {message}")


def load():
    """Load C++ module and initialise Vulkan engine. Returns success."""
    global _core, _engine

    if _engine is not None:
        return True

    # Ensure wheel-bundled DLLs are discoverable on Windows.
    try:
        from ..utils.dll_loader import add_wheel_dll_dirs
        add_wheel_dll_dirs("texturesynth_core")
        add_wheel_dll_dirs("texturesynth")
    except Exception:
        pass

    # Import texturesynth_core binary module.
    try:
        import texturesynth_core as core
        _core = core
        try:
            core.set_log_callback(_cpp_log_sink)
        except Exception:
            pass
    except ImportError as e:
        _tslog.error(f"ImportError: {e}")
        _tslog.error("Engine will not be available. Nodes still work as UI.")
        return False
    except Exception as e:
        _tslog.error(f"Core import exception: {e}")
        return False

    # Instantiate engine.
    try:
        eng = _core.Engine()
    except Exception as e:
        _tslog.error(f"Engine() constructor exception: {e}")
        _core = None
        return False

    # Initialize headless Vulkan context and shader directories.
    try:
        cache_dir = os.path.join(
            bpy.utils.user_resource('DATAFILES', path="texturesynth"),
            "shader_cache",
        )
        addon_root = os.path.dirname(os.path.dirname(__file__))
        nodes_dir  = os.path.join(addon_root, "shader_assets", "nodes")
        glsl_dir   = os.path.join(addon_root, "shader_assets", "glsl")

        os.makedirs(cache_dir, exist_ok=True)

        ok = eng.init(enable_validation=False, cache_dir=cache_dir, nodes_dir=nodes_dir, glsl_dir=glsl_dir)
        if not ok:
            _tslog.error(f"Engine.init() failed: {eng.last_error()}")
            _core = None
            return False
    except Exception as e:
        _tslog.error(f"Engine.init() exception: {e}")
        _core = None
        return False

    _engine = eng
    return True


def shutdown():
    """Destroy the Vulkan engine. Safe to call even if never loaded."""
    global _core, _engine
    if _engine is not None:
        try:
            _engine.shutdown()
        except Exception as e:
            _tslog.error(f"Engine shutdown exception: {e}")
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
