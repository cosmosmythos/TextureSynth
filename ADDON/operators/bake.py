"""Bake operator: push the graph to the engine, then iterate the output
node's named targets, render each, and write one bpy.data.images.Image
per target. The host (Blender) owns the format — no file I/O in the
engine. Users can right-click an image and Save As to persist.
"""
import time

import bpy
import numpy as np
from bpy.types import Operator
from ..core import cpp_module
from ..core import engine_bridge
from ..core import logging as _tslog
from ..nodes.specialized.output import TS_Output_Node


def _active_output_node(context):
    """Find the output node in the active TextureSynth node tree, or
    the first one in any tree if no editor is open."""
    tree = engine_bridge._find_node_tree()
    if tree is None:
        return None
    for n in tree.nodes:
        if n.bl_idname == 'TS_Output_Node':
            return n
    return None


def _collect_targets(output_node):
    """Returns [(source_node_id, name), ...] from the output node's
    targets collection. Skips rows with empty source_node string.
    A source_node of "0" is interpreted as "use the link feeding the
    Result input" (the live-preview source) — convenient default.
    """
    # The live preview source — the node feeding the 'Result' input.
    # Used when a target row has source_node == "0" or empty.
    preview_source = _preview_source_id(output_node)
    out = []
    for t in output_node.targets:
        raw = (t.source_node or "").strip()
        if not raw:
            # No source set: fall back to the preview source.
            if preview_source is None:
                continue
            sid = preview_source
        else:
            try:
                sid = int(raw, 0)  # accepts "0x..." or decimal
            except ValueError:
                _tslog.error(
                    f"target '{t.name}' has invalid source_node id: {raw!r}")
                continue
        out.append((sid, t.name or "Unnamed"))
    return out


def _preview_source_id(output_node):
    """Resolve the stable_id of the node feeding the Output node's
    'Result' input. Returns int, or None if not linked."""
    if not output_node.inputs:
        return None
    for sock in output_node.inputs:
        if sock.is_linked and sock.links:
            src = sock.links[0].from_node
            if src is not None and hasattr(src, "stable_id"):
                return int(src.stable_id())
    return None


class TEXTURESYNTH_OT_bake(Operator):
    bl_idname = "texturesynth.bake"
    bl_label  = "Bake TextureSynth"
    bl_description = ("Render all output targets to bpy.data.images. "
                      "Each target becomes a separate image named after its "
                      "'Name' field. Set the source node id for each target "
                      "in the output node's N-panel before baking.")

    def execute(self, context):
        engine = cpp_module.get_engine()
        if not cpp_module.is_loaded():
            self.report({"ERROR"}, "TextureSynth engine not loaded")
            return {"CANCELLED"}

        # 1. Find the output node and its targets.
        output_node = _active_output_node(context)
        if output_node is None:
            self.report({"ERROR"}, "No TextureSynth Output node found in the active tree")
            return {"CANCELLED"}
        targets = _collect_targets(output_node)
        if not targets:
            self.report({"WARNING"},
                        "No output targets set on the output node. "
                        "Set each target's 'Src' to a node ID, then click + if needed.")
            return {"CANCELLED"}

        # 2. Push the graph to the engine. submit_graph() already injects
        #    the output node's named targets into the Graph before
        #    set_graph(), so no manual target injection is needed here.
        if not engine_bridge.submit_graph():
            self.report({"ERROR"},
                        f"graph submit failed: {engine.last_error()}")
            return {"CANCELLED"}
        for _ in range(200):
            if engine.has_pipeline():
                break
            engine.poll_pending_compiles()
            time.sleep(0.01)
        if not engine.has_pipeline():
            self.report({"ERROR"}, "compile timed out")
            return {"CANCELLED"}

        # 3. Synchronous bake: returns list of {name, w, h, ndarray}.
        bakes = engine.bake()
        if not bakes:
            self.report({"ERROR"}, f"bake failed: {engine.last_error()}")
            return {"CANCELLED"}

        # 4. Write each bakes entry to bpy.data.images.
        for b in bakes:
            name = b["name"]
            w, h = int(b["width"]), int(b["height"])
            arr = np.asarray(b["pixels"], dtype=np.float32).reshape(h, w, 4)
            existing = bpy.data.images.get(name)
            if existing is not None and (existing.size[0] != w or existing.size[1] != h):
                # Size mismatch: drop and recreate.
                bpy.data.images.remove(existing)
                existing = None
            if existing is None:
                img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
            else:
                img = existing
            img.pixels.foreach_set(arr.astype(np.float32, copy=False).ravel())
            img.update()
            self.report({"INFO"},
                        f"Baked '{name}' {w}x{h} -> bpy.data.images['{name}']")

        return {"FINISHED"}


classes = (TEXTURESYNTH_OT_bake,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
