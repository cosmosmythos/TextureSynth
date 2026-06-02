"""
Logging helpers for the TextureSynth addon.

The C++ engine has its own log_info/log_warn/log_error pipeline (see
src/engine/Logging.hpp). The Python side installs a callback via
`texturesynth_core.set_log_callback(level, message)` and that callback
filters by level here, so the user only sees events they can act on.

Default level: WARNING (errors + warnings). Switch to DEBUG in the
addon prefs to get the full firehose.
"""
import logging
import bpy

LOGGER_NAME = "texturesynth"

# ── module-level logger ───────────────────────────────────────────────
_log = logging.getLogger(LOGGER_NAME)
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[TextureSynth] %(message)s"))
    _log.addHandler(_h)
_log.propagate = False


_LEVELS = ("ERROR", "WARNING", "INFO", "DEBUG")


def _prefs_level() -> str:
    """Read the current log_level from addon preferences. Safe to call
    before bpy.context is fully available (returns 'ERROR' as a safe default)."""
    try:
        addon = bpy.context.preferences.addons.get(__package__ or "texturesynth")
        if addon is None:
            return "ERROR"
        lvl = getattr(addon.preferences, "log_level", "ERROR")
        return lvl if lvl in _LEVELS else "ERROR"
    except Exception:
        return "ERROR"


def update_level():
    """Re-apply the level from the addon preferences."""
    _log.setLevel(getattr(logging, _prefs_level(), logging.ERROR))


def is_enabled_for(level_name: str) -> bool:
    """True if a message at `level_name` would actually be emitted given
    the user's current preference. Use to skip expensive string formatting
    on the hot path."""
    cur = getattr(logging, _prefs_level(), logging.ERROR)
    want = getattr(logging, level_name, logging.DEBUG)
    return want >= cur


# ── convenience helpers ───────────────────────────────────────────────
def debug(msg, *args, **kwargs):
    if is_enabled_for("DEBUG"):
        _log.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    if is_enabled_for("INFO"):
        _log.info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    if is_enabled_for("WARNING"):
        _log.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    if is_enabled_for("ERROR"):
        _log.error(msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
    """Log at ERROR with traceback. Use only at exception sites."""
    if is_enabled_for("ERROR"):
        _log.error(msg, *args, exc_info=exc_info, **kwargs)
