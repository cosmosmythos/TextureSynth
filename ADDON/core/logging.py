"""Logging helpers for the TextureSynth addon.
Filters messages received from the C++ texturesynth_core log sink based on preferences."""
import logging
import bpy

LOGGER_NAME = "texturesynth"

_log = logging.getLogger(LOGGER_NAME)
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[TextureSynth] %(message)s"))
    _log.addHandler(_h)
_log.propagate = False

_LEVELS = ("ERROR", "WARNING", "INFO", "DEBUG")


def _prefs_level() -> str:
    """Read log_level from preferences, defaulting to ERROR if context is not ready."""
    try:
        addon = bpy.context.preferences.addons.get(__package__ or "texturesynth")
        if addon is None:
            return "ERROR"
        prefs_level = getattr(addon.preferences, "log_level", "ERROR")
        return prefs_level if prefs_level in _LEVELS else "ERROR"
    except Exception:
        return "ERROR"


def update_level():
    """Re-apply the level from the addon preferences."""
    _log.setLevel(getattr(logging, _prefs_level(), logging.ERROR))


def is_enabled_for(level_name: str) -> bool:
    """Check if a log level is enabled to avoid formatting overhead."""
    current_level = getattr(logging, _prefs_level(), logging.ERROR)
    wanted_level = getattr(logging, level_name, logging.DEBUG)
    return wanted_level >= current_level


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
