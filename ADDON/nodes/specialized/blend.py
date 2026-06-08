"""Blend node — enum dropdown of all Photoshop modes."""
import bpy
from ..base import TextureSynthNode
from ._common import update_param

SV_TYPE = "blend"

# IMPORTANT: order MUST match blend.glsl mode integers
BLEND_MODES = [
    ('MIX',          "Mix",           ""), ('ADD',          "Add",           ""),
    ('MULTIPLY',     "Multiply",      ""), ('SCREEN',       "Screen",        ""),
    ('OVERLAY',      "Overlay",       ""), ('DIFFERENCE',   "Difference",    ""),
    ('DARKEN',       "Darken",        ""), ('LIGHTEN',      "Lighten",       ""),
    ('COLOR_BURN',   "Color Burn",    ""), ('COLOR_DODGE',  "Color Dodge",   ""),
    ('LINEAR_BURN',  "Linear Burn",   ""), ('LINEAR_DODGE', "Linear Dodge",  ""),
    ('LINEAR_LIGHT', "Linear Light",  ""), ('VIVID_LIGHT',  "Vivid Light",   ""),
    ('PIN_LIGHT',    "Pin Light",     ""), ('HARD_LIGHT',   "Hard Light",    ""),
    ('SOFT_LIGHT',   "Soft Light",    ""), ('HARD_MIX',     "Hard Mix",      ""),
    ('EXCLUSION',    "Exclusion",     ""), ('SUBTRACT',     "Subtract",      ""),
    ('AVERAGE',      "Average",       ""), ('NEGATION',     "Negation",      ""),
    ('REFLECT',      "Reflect",       ""), ('GLOW',         "Glow",          ""),
    ('HARMONY',      "Harmony",       ""), ('HUE',          "Hue",           ""),
    ('SATURATION',   "Saturation",    ""), ('COLOR',        "Color",         ""),
    ('LUMINOSITY',   "Luminosity",    ""),
]
_MODE_INDEX = {k: i for i, (k, _, _) in enumerate(BLEND_MODES)}


class TS_Blend_Node(TextureSynthNode):
    bl_idname = 'TS_Blend_Node'
    bl_label  = 'Blend'
    sv_type   = SV_TYPE
    ts_category = 'BLEND'
    supports_format_override = False

    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[(k, n, d) for (k, n, d) in BLEND_MODES],
        default='MIX',
        update=update_param,
    )
    factor: bpy.props.FloatProperty(
        name="Factor", min=0.0, max=1.0, default=1.0,
        precision=3, update=update_param,
    )
    mask: bpy.props.FloatProperty(
        name="Mask", min=0.0, max=1.0, default=1.0,
        precision=3, update=update_param,
    )

    def init(self, context):
        super().init(context)
        self.inputs.new('TS_DefaultSocketType', "A")
        self.inputs.new('TS_DefaultSocketType', "B")
        self.inputs.new('TS_DefaultSocketType', "Mask")
        self.outputs.new('TS_DefaultSocketType', "")

    def draw_buttons(self, context, layout):
        self.draw_error_ui(layout)
        self.draw_format_override_ui(layout)
        layout.prop(self, "mode", text="")
        layout.prop(self, "factor", slider=True)
        if "Mask" in self.inputs and not self.inputs["Mask"].is_linked:
            layout.prop(self, "mask", slider=True)

    # JSON order: [factor, mode, mask]
    def get_parameters(self):
        return [float(self.factor), float(_MODE_INDEX[self.mode]),
                float(self.mask)]

    def get_named_parameters(self):
        # Must match JSON manifest param order/names exactly.
        return {"factor": float(self.factor),
                "mode":   float(_MODE_INDEX[self.mode]),
                "mask":   float(self.mask)}


NODE_CLASS = TS_Blend_Node
