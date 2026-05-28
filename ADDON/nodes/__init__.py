"""
Node definitions for TextureSynth.
"""

from . import tree
from . import categories
from . import factory
from ..core import cpp_module

def register():
    tree.register()
    
    # Generate nodes dynamically before registering UI
    factory.generate_node_classes(cpp_module)
    factory.register()
    
    categories.register()

def unregister():
    categories.unregister()
    factory.unregister()
    tree.unregister()
