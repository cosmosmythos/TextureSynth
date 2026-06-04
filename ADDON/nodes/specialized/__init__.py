"""
Specialized (hand-written) TextureSynth nodes.

Each module in this package defines exactly ONE node class that the
auto-generator in factory.py cannot or should not produce — typically
because the node needs:

  - a custom Blender UI widget (color picker, curve, ramp, image slot)
  - mode-switched parameter layouts
  - non-trivial get_parameters() logic that maps UI state to JSON param order

CONTRACT for a specialized module:
  - Defines `SV_TYPE: str`  — the JSON node id this module owns
                              (or None for UI-only nodes like Output)
  - Defines `NODE_CLASS`    — a class subclassing TextureSynthNode

Discovery:
  Modules are imported lazily by `iter_specialized_modules()`.
  The set of owned sv_types is exposed via `specialized_sv_types()`,
  which factory.py uses as the skip-set.
"""
from __future__ import annotations
import importlib
from typing import Iterator


# Modules listed here are loaded in order. Add a line per new specialized node.
_MODULE_NAMES = (
    "color_const",
    "blend",
    "image",
    "output",
)


def iter_specialized_modules() -> Iterator[object]:
    """Yield each specialized module, importing it on demand."""
    pkg = __name__  # "...nodes.specialized"
    for name in _MODULE_NAMES:
        yield importlib.import_module(f"{pkg}.{name}")


def specialized_sv_types() -> set[str]:
    """sv_type strings owned by specialized modules.
    Returned to factory.py as the skip-set. UI-only nodes (SV_TYPE=None)
    are excluded — they have no JSON counterpart to skip."""
    result = set()
    for mod in iter_specialized_modules():
        sv = getattr(mod, "SV_TYPE", None)
        if sv:
            result.add(sv)
    return result


def collect_node_classes() -> list:
    """Return every NODE_CLASS from every specialized module, in declaration order."""
    return [mod.NODE_CLASS for mod in iter_specialized_modules()
            if hasattr(mod, "NODE_CLASS")]