"""Operators for adding / removing / selecting rows in the output
node's targets collection. Bound to the +/- buttons in
TS_Output_Node.draw_buttons.
"""
import bpy
from bpy.types import Operator
from ..nodes.specialized.output import TS_Output_Node


class TEXTURESYNTH_OT_output_target_add(Operator):
    bl_idname = "texturesynth.output_target_add"
    bl_label  = "Add Output Target"
    bl_description = "Add a new named output target. The new row becomes the active target."

    def execute(self, context):
        node = context.active_node
        if not isinstance(node, TS_Output_Node):
            return {"CANCELLED"}
        t = node.targets.add()
        # Name suggestions cycle through the common PBR slots, then fall
        # back to "Target N" once exhausted.
        pbr_defaults = (
            "Base Color", "Normal", "Roughness", "Metallic", "Height",
            "Ambient Occlusion", "Emissive", "Opacity", "Displacement",
        )
        if len(node.targets) - 1 < len(pbr_defaults):
            t.name = pbr_defaults[len(node.targets) - 1]
        else:
            t.name = f"Target {len(node.targets)}"
        t.source_node = ""
        node.active_target_index = len(node.targets) - 1
        return {"FINISHED"}


class TEXTURESYNTH_OT_output_target_remove(Operator):
    bl_idname = "texturesynth.output_target_remove"
    bl_label  = "Remove Output Target"
    bl_description = "Remove the currently active target."

    def execute(self, context):
        node = context.active_node
        if not isinstance(node, TS_Output_Node) or not node.targets:
            return {"CANCELLED"}
        idx = max(0, min(node.active_target_index, len(node.targets) - 1))
        node.targets.remove(idx)
        node.active_target_index = max(0, min(idx, len(node.targets) - 1))
        return {"FINISHED"}


class TEXTURESYNTH_OT_output_target_select(Operator):
    bl_idname = "texturesynth.output_target_select"
    bl_label  = "Select Output Target"
    bl_description = "Make this target the active one."
    index: bpy.props.IntProperty()

    def execute(self, context):
        node = context.active_node
        if isinstance(node, TS_Output_Node):
            node.active_target_index = self.index
        return {"FINISHED"}


classes = (
    TEXTURESYNTH_OT_output_target_add,
    TEXTURESYNTH_OT_output_target_remove,
    TEXTURESYNTH_OT_output_target_select,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
