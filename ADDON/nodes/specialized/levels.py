"""Levels node — per-channel independent input/output levels."""
import bpy
from ..base import TextureSynthNode
from ._common import update_param

SV_TYPE = "levels"

_CHANNELS = [
    ('L', "L", "Luminance"),
    ('R', "R", "Red"),
    ('G', "G", "Green"),
    ('B', "B", "Blue"),
    ('A', "A", "Alpha"),
]
_CHANNEL_INDEX = {k: i for i, (k, _, _) in enumerate(_CHANNELS)}

_PARAMS_PER_CH = 5


def _ch_props(channel):
    prefix = channel.lower()
    return {
        f"in_low_{prefix}": bpy.props.FloatProperty(
            name="In Min", default=0.0,
            min=-100000.0, max=100000.0, soft_min=0.0, soft_max=1.0,
            update=update_param),
        f"in_mid_{prefix}": bpy.props.FloatProperty(
            name="In Bias", default=0.5,
            min=0.0, max=1.0, soft_min=0.0, soft_max=1.0,
            subtype='FACTOR',
            update=update_param),
        f"in_high_{prefix}": bpy.props.FloatProperty(
            name="In Max", default=1.0,
            min=-100000.0, max=100000.0, soft_min=0.0, soft_max=1.0,
            update=update_param),
        f"out_low_{prefix}": bpy.props.FloatProperty(
            name="Out Min", default=0.0,
            min=-100000.0, max=100000.0, soft_min=0.0, soft_max=1.0,
            update=update_param),
        f"out_high_{prefix}": bpy.props.FloatProperty(
            name="Out Max", default=1.0,
            min=-100000.0, max=100000.0, soft_min=0.0, soft_max=1.0,
            update=update_param),
    }


_all_props = {}
for _ch, _lbl, _desc in _CHANNELS:
    _all_props.update(_ch_props(_ch))


class TS_Levels_Node(TextureSynthNode):
    bl_idname = 'TS_Levels_Node'
    bl_label = 'Levels'
    sv_type = SV_TYPE
    ts_category = 'COLOR'
    supports_format_override = False

    in_low_l: _all_props['in_low_l']
    in_mid_l: _all_props['in_mid_l']
    in_high_l: _all_props['in_high_l']
    out_low_l: _all_props['out_low_l']
    out_high_l: _all_props['out_high_l']

    in_low_r: _all_props['in_low_r']
    in_mid_r: _all_props['in_mid_r']
    in_high_r: _all_props['in_high_r']
    out_low_r: _all_props['out_low_r']
    out_high_r: _all_props['out_high_r']

    in_low_g: _all_props['in_low_g']
    in_mid_g: _all_props['in_mid_g']
    in_high_g: _all_props['in_high_g']
    out_low_g: _all_props['out_low_g']
    out_high_g: _all_props['out_high_g']

    in_low_b: _all_props['in_low_b']
    in_mid_b: _all_props['in_mid_b']
    in_high_b: _all_props['in_high_b']
    out_low_b: _all_props['out_low_b']
    out_high_b: _all_props['out_high_b']

    in_low_a: _all_props['in_low_a']
    in_mid_a: _all_props['in_mid_a']
    in_high_a: _all_props['in_high_a']
    out_low_a: _all_props['out_low_a']
    out_high_a: _all_props['out_high_a']

    channel_mode: bpy.props.EnumProperty(
        name="Channel",
        items=_CHANNELS,
        default='L',
        update=update_param)

    def init(self, context):
        super().init(context)
        self.inputs.new('TS_DefaultSocketType', "")
        self.outputs.new('TS_DefaultSocketType', "")

    def draw_buttons(self, context, layout):
        self.draw_error_ui(layout)
        layout.prop(self, "channel_mode", expand=True)
        ch = self.channel_mode.lower()
        layout.prop(self, f"in_low_{ch}")
        layout.prop(self, f"in_mid_{ch}")
        layout.prop(self, f"in_high_{ch}")
        layout.prop(self, f"out_low_{ch}")
        layout.prop(self, f"out_high_{ch}")

    def get_parameters(self):
        params = []
        for prefix in ('l', 'r', 'g', 'b', 'a'):
            params.append(float(getattr(self, f"in_low_{prefix}")))
            params.append(float(getattr(self, f"in_mid_{prefix}")))
            params.append(float(getattr(self, f"in_high_{prefix}")))
            params.append(float(getattr(self, f"out_low_{prefix}")))
            params.append(float(getattr(self, f"out_high_{prefix}")))
        params.append(float(_CHANNEL_INDEX[self.channel_mode]))
        return params

    def get_named_parameters(self):
        return None


NODE_CLASS = TS_Levels_Node
