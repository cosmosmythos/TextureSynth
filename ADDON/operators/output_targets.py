"""Operators to add or remove input sockets on the Output node."""
import bpy
from bpy.types import Operator
from ..nodes.specialized.output import TS_Output_Node
from ..core.evaluation import request_topology_update

# Target names rotation to select the first unused default PBR target name.
_PBR_DEFAULTS = (
    "Base Color", "Normal", "Roughness", "Metallic", "Height",
    "Ambient Occlusion", "Emissive", "Opacity", "Displacement",
    "Specular", "Glossiness", "Subsurface",
)


def _next_default_name(node):
    existing = {s.name for s in node.inputs}
    for n in _PBR_DEFAULTS:
        if n not in existing:
            return n
    return f"Target {len(node.inputs) + 1}"


class TEXTURESYNTH_OT_output_target_add(Operator):
    bl_idname = "texturesynth.output_target_add"
    bl_label  = "Add Bake Target"
    bl_description = "Add a new input socket representing a bake target to the Output node"

    def execute(self, context):
        node = context.active_node
        if not isinstance(node, TS_Output_Node):
            return {"CANCELLED"}
        node.inputs.new('TS_BakeTargetSocketType', _next_default_name(node))
        request_topology_update()
        return {"FINISHED"}


class TEXTURESYNTH_OT_output_target_remove(Operator):
    bl_idname = "texturesynth.output_target_remove"
    bl_label  = "Remove Last Bake Target"
    bl_description = "Remove the last input socket from the Output node"

    @classmethod
    def poll(cls, context):
        node = context.active_node
        return isinstance(node, TS_Output_Node) and len(node.inputs) > 0

    def execute(self, context):
        node = context.active_node
        if not isinstance(node, TS_Output_Node) or not node.inputs:
            return {"CANCELLED"}
        node.inputs.remove(node.inputs[-1])
        request_topology_update()
        return {"FINISHED"}


classes = (
    TEXTURESYNTH_OT_output_target_add,
    TEXTURESYNTH_OT_output_target_remove,
)

register, unregister = bpy.utils.register_classes_factory(classes)
