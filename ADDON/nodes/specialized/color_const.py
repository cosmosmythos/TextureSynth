"""Color Input node supporting Grayscale and RGBA modes."""
import bpy
from ..base import TextureSynthNode
from ._common import update_param

SV_TYPE = "color_const"


class TS_ColorConst_Node(TextureSynthNode):
    bl_idname = 'TS_ColorConst_Node'
    bl_label  = 'Color'
    sv_type   = SV_TYPE
    ts_category = 'INPUT'
    supports_format_override = True

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
        super().init(context)
        self.outputs.new('TS_DefaultSocketType', "")

    def draw_buttons(self, context, layout):
        self.draw_error_ui(layout)
        self.draw_format_override_ui(layout)
        layout.prop(self, "mode", expand=True)
        if self.mode == 'RGBA':
            layout.template_color_picker(self, "color_data", value_slider=True)
            layout.prop(self, "color_data", text="")
        else:
            layout.prop(self, "gray_value", slider=True)

    def get_parameters(self):
        # Param order: [mode, r, g, b, a]
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
