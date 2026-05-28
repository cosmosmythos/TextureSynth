"""
Core — C++ engine loading and evaluation loop.
"""

from . import cpp_module
from . import evaluation


def register():
    cpp_module.load()          # attempt pyd import (non-fatal if fails)
    evaluation.register()      # start evaluation timer


def unregister():
    evaluation.unregister()    # stop timer
    cpp_module.shutdown()      # destroy Vulkan engine
