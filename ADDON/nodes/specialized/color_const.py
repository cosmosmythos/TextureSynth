"""
Color Input node.

Backed by shader_assets/nodes/color_const.node.json + color_const.glsl.

JSON param order (MUST be matched by get_parameters):
    [mode, r, g, b, a]

Modes:
    mode = 0.0  -> Grayscale  : output (v, v, v, 1)
    mode = 1.0  -> RGBA       : output (r, g, b, a)
"""
import bpy
from ..base import TextureSynthNode
from ._common import update_param


SV_TYPE = "color_const"   # matches color_const.node.json `id`


class TS_ColorConst_Node(TextureSynthNode):
    bl_idname = 'TS_ColorConst_Node'
    bl_label  = 'Color'
    sv_type   = SV_TYPE

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('RGBA', "Color",     "RGBA color picker"),
            ('GRAY', "Grayscale", "Single grayscale value"),
        ],
        default='RGBA',
        update=update_param,
    )
    color_data: bpy.props.FloatVectorProperty(
        name="Color",
        size=4,
        subtype='COLOR',
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        update=update_param,
    )
    gray_value: bpy.props.FloatProperty(
        name="Value",
        min=0.0, max=1.0,
        default=1.0,
        precision=3,
        update=update_param,
    )

    def init(self, context):
        super().init(context)              # assigns ts_uuid
        self.outputs.new('TextureSynthSocketType', "")

    def draw_buttons(self, context, layout):
        self.draw_error_ui(layout)
        layout.prop(self, "mode", expand=True)
        if self.mode == 'RGBA':
            layout.template_color_picker(self, "color_data", value_slider=True)
            layout.prop(self, "color_data", text="")
        else:
            layout.prop(self, "gray_value", slider=True)

    def get_parameters(self):
        # Order: [mode, r, g, b, a]
        if self.mode == 'RGBA':
            c = self.color_data
            return [1.0, float(c[0]), float(c[1]), float(c[2]), float(c[3])]
        v = float(self.gray_value)
        return [0.0, v, v, v, 1.0]

    def get_named_parameters(self):
        if self.mode == 'RGBA':
            c = self.color_data
            return {"mode": 1.0, "r": float(c[0]), "g": float(c[1]),
                    "b": float(c[2]), "a": float(c[3])}
        v = float(self.gray_value)
        return {"mode": 0.0, "r": v, "g": v, "b": v, "a": 1.0}


NODE_CLASS = TS_ColorConst_Node