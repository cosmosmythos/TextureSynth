"""Output node serving as the final bake sink for generated texture channels."""
import bpy
from ..base import TextureSynthNode
from ..tree import TS_TextureSocket


class TS_BakeTargetSocket(TS_TextureSocket):
    """Input socket representing a target bake channel (e.g. Base Color, Normal)."""
    bl_idname = 'TS_BakeTargetSocketType'
    bl_label = "Bake Target"

    @classmethod
    def draw_color_simple(cls):
        return (0.18, 0.18, 0.18, 1.0)

    def draw(self, context, layout, node, text):
        if self.is_linked:
            layout.label(text=self.name)
        else:
            layout.prop(self, "name", text=text)


class TS_Output_Node(TextureSynthNode):
    """The bake sink node. Dynamic input sockets; one socket = one bake target."""
    bl_idname = 'TS_Output_Node'
    bl_label  = 'Output'
    sv_type   = None
    ts_category = 'OUTPUT'

    def init(self, context):
        super().init(context)
        # Only seed default sockets when creating a new node instance.
        if self.inputs:
            return
        # Seed with conventional PBR triad: Base Color, Normal, Roughness.
        for default_name in ("Base Color", "Normal", "Roughness"):
            self.inputs.new('TS_BakeTargetSocketType', default_name)

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.operator("texturesynth.output_target_add", text="", icon="ADD")
        row.operator("texturesynth.output_target_remove", text="", icon="REMOVE")
        layout.separator()
        layout.operator("texturesynth.bake", icon='RENDER_STILL', text="Bake / Export")


NODE_CLASS = TS_Output_Node

# Sockets to register before registration of Output node.
SOCKET_CLASSES = (TS_BakeTargetSocket,)

PROPERTY_GROUPS = ()
