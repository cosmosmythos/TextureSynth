"""
Operators for TextureSynth.
"""

from . import update, connect


def register():
    update.register()
    connect.register()


def unregister():
    connect.unregister()
    update.unregister()
