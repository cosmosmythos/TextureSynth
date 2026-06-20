"""Specialized hand-written nodes for TextureSynth requiring custom UI layouts, properties, or bindings."""
from __future__ import annotations
import importlib
from typing import Iterator

_MODULE_NAMES = (
    "color_const",
    "blend",
    "image",
    "levels",
    "output",
)


def iter_specialized_modules() -> Iterator[object]:
    """Yield each specialized module, importing it on demand."""
    pkg = __name__
    for name in _MODULE_NAMES:
        yield importlib.import_module(f"{pkg}.{name}")


def specialized_sv_types() -> set[str]:
    """sv_type strings owned by specialized modules, used as the generator skip-set."""
    result = set()
    for mod in iter_specialized_modules():
        sv = getattr(mod, "SV_TYPE", None)
        if sv:
            result.add(sv)
    return result


def specialized_property_groups() -> list:
    """PropertyGroup classes that must be registered before node classes referencing them."""
    result = []
    for mod in iter_specialized_modules():
        for cls in getattr(mod, "PROPERTY_GROUPS", ()):
            result.append(cls)
    return result


def collect_node_classes() -> list:
    """Return every NODE_CLASS from every specialized module, in declaration order."""
    return [mod.NODE_CLASS for mod in iter_specialized_modules()
            if hasattr(mod, "NODE_CLASS")]


def collect_socket_classes() -> list:
    """Return every NodeSocket subclass exported by a specialized module."""
    result = []
    for mod in iter_specialized_modules():
        for cls in getattr(mod, "SOCKET_CLASSES", ()):
            result.append(cls)
    return result