"""Blend node — 2-input combiner. A = foreground, B = background."""
import bpy
from ..base import TextureSynthNode
from ._common import update_param

SV_TYPE = "blend"

# order MUST match blend.glsl mode integers.
BLEND_MODES = [
    ('MIX',          "Mix",          ""),
    ('ADD',          "Add",          ""),
    ('SUBTRACT',     "Subtract",     ""),
    ('MULTIPLY',     "Multiply",     ""),
    ('MIN',          "Min (Darken)", ""),
    ('MAX',          "Max (Lighten)",""),
    ('AVERAGE',      "Average",      ""),
    ('COLOR_BURN',   "Color Burn",   ""),
    ('OVERLAY',      "Overlay",      ""),
    ('SCREEN',       "Screen",       ""),
    ('COLOR_DODGE',  "Color Dodge",  ""),
    ('SOFT_LIGHT',   "Soft Light",   ""),
    ('HARD_LIGHT',   "Hard Light",   ""),
    ('DIVIDE',       "Divide",       ""),
    ('DIFFERENCE',   "Difference",   ""),
    ('EXCLUSION',    "Exclusion",    ""),
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
    mask: bpy.props.FloatProperty(
        name="Mask", default=1.0, min=0.0, max=1.0,
        subtype='FACTOR', update=update_param,
    )

    def init(self, context):
        super().init(context)
        self.inputs.new('TS_DefaultSocketType', "Mask")
        self.inputs.new('TS_DefaultSocketType', "A")  # foreground (top, mask=1)
        self.inputs.new('TS_DefaultSocketType', "B")  # background (base, mask=0)
        self.outputs.new('TS_DefaultSocketType', "")

    def draw_buttons(self, context, layout):
        self.draw_error_ui(layout)
        self.draw_format_override_ui(layout)
        layout.prop(self, "mode", text="")

    # SSBO order: [mode_param, mask_default]
    def get_parameters(self):
        return [float(_MODE_INDEX[self.mode]), float(self.mask)]

    def get_named_parameters(self):
        return None  # mask is a float input; use get_parameters() for full SSBO write


NODE_CLASS = TS_Blend_Node
