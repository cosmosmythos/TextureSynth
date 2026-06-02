"""
Addon preferences for TextureSynth.

Currently exposes a single `log_level` enum. The C++ engine routes its
log_info/log_warn/log_error messages through a Python callback (see
cpp_module._cpp_log_sink); that callback reads the level from here and
drops events below the threshold.

Default = WARNING. Set to DEBUG in the addon prefs to see the full
firehose (PassPlan compile/install, ResourceManager allocations, etc.)
when diagnosing an issue from a user report.
"""
import bpy
from bpy.types import AddonPreferences
from bpy.props import EnumProperty, BoolProperty

from . import logging as _tslog

_LOG_LEVELS = [
    ("ERROR",   "Errors only",        "Errors only. C++ warnings, info, and Python debug/info are silenced.", 0),
    ("WARNING", "Errors + Warnings",  "Default. Errors and warnings only.",                                     1),
    ("INFO",    "Info",               "Adds high-level events (e.g. submitted graph generation=N).",            2),
    ("DEBUG",   "Debug",              "Full firehose. Switch to this and reproduce the issue to gather logs.",  3),
]


def _on_log_level_change(self, context):
    _tslog.update_level()


class TextureSynthPreferences(AddonPreferences):
    bl_idname = __package__ or "texturesynth"

    log_level: EnumProperty(
        name="Log level",
        description="Minimum event severity to print. More-severe messages are also printed.",
        items=_LOG_LEVELS,
        default="WARNING",
        update=_on_log_level_change,
    )

    log_tracebacks: BoolProperty(
        name="Log exception tracebacks",
        description="Attach Python tracebacks to error logs (useful for bug reports).",
        default=False,
    )

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Logging")
        box.prop(self, "log_level", text="Level")
        box.prop(self, "log_tracebacks")


def register():
    bpy.utils.register_class(TextureSynthPreferences)
    _tslog.update_level()


def unregister():
    bpy.utils.unregister_class(TextureSynthPreferences)
