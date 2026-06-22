"""Idempotent wrappers around registration functions to tolerate classes already in bpy.types."""
import bpy


def register_class(cls):
    """Register class if not already in bpy.types, tolerating registration races."""
    if cls.is_registered:
        return
    try:
        bpy.utils.register_class(cls)
    except ValueError:
        # Stale-state race: tolerate class registered concurrently.
        pass


def unregister_class(cls):
    """Unregister class if currently in bpy.types."""
    if not cls.is_registered:
        return
    try:
        bpy.utils.unregister_class(cls)
    except (ValueError, RuntimeError):
        # Tolerate class registered by stale imports but no longer active
        pass
