"""Addon preferences for TextureSynth.
Exposes logging levels and debug settings."""
import bpy
from bpy.types import AddonPreferences
from bpy.props import EnumProperty, BoolProperty

from . import logging as _tslog
from ..utils.rna import register_class, unregister_class

_LOG_LEVELS = [
    ("ERROR",   "Errors only",        "",                          0),
    ("WARNING", "Errors + Warnings",  "Default",                   1),
    ("INFO",    "Info",               "Adds submission events",    2),
    ("DEBUG",   "Debug",              "Verbose — use for debugging", 3),
]


def _on_log_level_change(self, context):
    _tslog.update_level()


class TextureSynthPreferences(AddonPreferences):
    bl_idname = __package__ or "texturesynth"

    log_level: EnumProperty(
        name="Log level",
        description="Minimum severity to print",
        items=_LOG_LEVELS,
        default="WARNING",
        update=_on_log_level_change,
    )

    log_tracebacks: BoolProperty(
        name="Log tracebacks",
        description="Include tracebacks in error logs",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Logging")
        box.prop(self, "log_level", text="Level")
        box.prop(self, "log_tracebacks")


def register():
    register_class(TextureSynthPreferences)
    _tslog.update_level()


def unregister():
    unregister_class(TextureSynthPreferences)
