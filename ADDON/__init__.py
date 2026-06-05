"""
TextureSynth — Node-based Vulkan Procedural Texture Generator
"""

from .core import cpp_module, evaluation, preferences
from . import nodes
from . import operators
from . import panels


def register():
    # Register tree first to avoid console noise from SpaceNodeEditor.tree_type mismatch.
    nodes.tree.register()

    # Preferences must load before the C++ log sink starts reading the level.
    preferences.register()

    # Load C++ engine before generating dynamic node classes.
    cpp_module.load()

    # Register remaining RNA/UI classes.
    nodes.register()
    operators.register()
    panels.register()

    # Start evaluation only after all node classes exist.
    evaluation.register()


def unregister():
    # Stop evaluation loop before tearing down classes.
    evaluation.unregister()

    panels.unregister()
    operators.unregister()
    nodes.unregister()

    # Shut down Vulkan engine and clean up preferences.
    cpp_module.shutdown()
    preferences.unregister()