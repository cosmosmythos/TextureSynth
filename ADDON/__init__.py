"""
TextureSynth — Node-based Vulkan Procedural Texture Generator
"""

from .core import cpp_module, evaluation, preferences
from . import nodes
from . import operators
from . import panels


def register():
    # Preferences must register first so the C++ log sink (installed
    # below) can read the user's preferred level.
    preferences.register()

    # Load C++ engine first so dynamic node classes can be generated.
    cpp_module.load()

    # Register RNA/UI classes before starting the timer.
    nodes.register()
    operators.register()
    panels.register()

    # Start evaluation only after all node classes exist.
    evaluation.register()


def unregister():
    # Stop timer before unregistering node classes.
    evaluation.unregister()

    panels.unregister()
    operators.unregister()
    nodes.unregister()

    cpp_module.shutdown()

    preferences.unregister()