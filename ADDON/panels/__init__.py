"""
UI Panels for TextureSynth.
"""

from . import sidebar


def register():
    sidebar.register()


def unregister():
    sidebar.unregister()
