"""
Operators for TextureSynth.
"""

from . import update, connect, output_targets, bake


def register():
    update.register()
    connect.register()
    output_targets.register()
    bake.register()


def unregister():
    bake.unregister()
    output_targets.unregister()
    connect.unregister()
    update.unregister()
