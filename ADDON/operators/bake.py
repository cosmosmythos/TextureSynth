"""Bake operator: pushes the graph to the engine and renders target channels to Blender images."""
import time

import bpy
import numpy as np
from bpy.types import Operator
from ..core import cpp_module
from ..core import engine_bridge
from ..core import logging as _tslog
from ..nodes.specialized.output import TS_Output_Node


def _active_output_node(context):
    """Find active node editor's Output node, or fallback to any tree."""
    tree = engine_bridge._find_node_tree()
    if tree is None:
        return None
    for n in tree.nodes:
        if n.bl_idname == 'TS_Output_Node':
            return n
    return None


def _bake_resolution():
    """Return (w, h) tuple using the sidebar settings."""
    res = int(getattr(bpy.context.scene, "texturesynth_resolution", 512))
    return res, res


def _create_black_image(name, w, h):
    """Create or replace a black float-buffer image for unlinked bake targets."""
    existing = bpy.data.images.get(name)
    if existing is not None and (existing.size[0] != w or existing.size[1] != h):
        bpy.data.images.remove(existing)
        existing = None
    if existing is None:
        img = bpy.data.images.new(name, w, h, alpha=True, float_buffer=True)
    else:
        img = existing
    zeros = np.zeros((h, w, 4), dtype=np.float32)
    img.pixels.foreach_set(zeros.ravel())
    img.update()
    return img


class TEXTURESYNTH_OT_bake(Operator):
    bl_idname = "texturesynth.bake"
    bl_label  = "Bake TextureSynth"
    bl_description = "Bake connected targets to Blender images"

    def execute(self, context):
        engine = cpp_module.get_engine()
        if not cpp_module.is_loaded():
            self.report({"ERROR"}, "TextureSynth not loaded")
            return {"CANCELLED"}

        output_node = _active_output_node(context)
        if output_node is None:
            self.report({"ERROR"}, "Output node not found in the active tree")
            return {"CANCELLED"}
        if not output_node.inputs:
            self.report({"WARNING"},
                        "No bake targets. Click + to add some.")
            return {"CANCELLED"}

        targets = []
        for sock in output_node.inputs:
            name = sock.name or "Unnamed"
            if sock.is_linked and sock.links:
                src = sock.links[0].from_node
                if src is not None and hasattr(src, "stable_id"):
                    targets.append((int(src.stable_id()), name, sock))
                    continue
            targets.append((None, name, sock))

        # Push the graph to the engine.
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

        # Synchronously trigger bake on Vulkan engine.
        bakes_by_name = {}
        if any(t[0] is not None for t in targets):
            try:
                bakes = engine.bake()
            except Exception as e:
                self.report({"ERROR"}, f"bake failed: {e}")
                return {"CANCELLED"}
            for b in bakes:
                bakes_by_name[b["name"]] = b

        # Write pixel buffers back to Blender images.
        w, h = _bake_resolution()
        written = 0
        empty = 0
        for src_id, name, sock in targets:
            if src_id is not None and name in bakes_by_name:
                b = bakes_by_name[name]
                bw, bh = int(b["width"]), int(b["height"])
                arr = np.asarray(b["pixels"], dtype=np.float32).reshape(bh, bw, 4)
                existing = bpy.data.images.get(name)
                if existing is not None and (existing.size[0] != bw or existing.size[1] != bh):
                    bpy.data.images.remove(existing)
                    existing = None
                if existing is None:
                    img = bpy.data.images.new(name, bw, bh, alpha=True, float_buffer=True)
                else:
                    img = existing
                img.pixels.foreach_set(arr.astype(np.float32, copy=False).ravel())
                img.update()
                written += 1
            else:
                _create_black_image(name, w, h)
                empty += 1
        self.report({"INFO"},
                    f"Baked {written} target(s), + "
                    f"{empty} empty target(s) (unlinked sockets)")
        return {"FINISHED"}


classes = (TEXTURESYNTH_OT_bake,)

register, unregister = bpy.utils.register_classes_factory(classes)
