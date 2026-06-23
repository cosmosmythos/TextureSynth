"""Levels node"""

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
_CHANNEL_INDEX = {key: i for i, (key, _, _) in enumerate(_CHANNELS)}

_FIELDS = ('in_low', 'in_mid', 'in_high', 'out_low', 'out_high')


def _mirror_to_rgb(field):
    """Return an update callback that copies field_l into field_r/g/b."""
    def update(self, context):
        val = getattr(self, f"{field}_l")
        for ch in ('r', 'g', 'b'):
            setattr(self, f"{field}_{ch}", val)
        update_param(self, context)
    return update


def _channel_props(channel):
    """Build FloatProperty kwargs for all five fields"""
    prefix = channel.lower()
    is_l = channel == 'L'
    make_update = _mirror_to_rgb if is_l else lambda f: update_param

    specs = [
        dict(name="In Min",  default=0.0, min=-100000.0, max=100000.0,
             soft_min=0.0, soft_max=1.0),
        dict(name="In Bias", default=0.5, min=0.0, max=1.0,
             soft_min=0.0, soft_max=1.0, subtype='FACTOR'),
        dict(name="In Max",  default=1.0, min=-100000.0, max=100000.0,
             soft_min=0.0, soft_max=1.0),
        dict(name="Out Min", default=0.0, min=-100000.0, max=100000.0,
             soft_min=0.0, soft_max=1.0),
        dict(name="Out Max", default=1.0, min=-100000.0, max=100000.0,
             soft_min=0.0, soft_max=1.0),
    ]
    return {
        f"{field}_{prefix}": bpy.props.FloatProperty(**spec, update=make_update(field))
        for field, spec in zip(_FIELDS, specs)
    }


# Collect all channel properties into a single flat dict.
_all_props = {}
for _ch, _lbl, _desc in _CHANNELS:
    _all_props.update(_channel_props(_ch))


class TS_Levels_Node(TextureSynthNode):
    bl_idname = 'TS_Levels_Node'
    bl_label = 'Levels'
    sv_type = SV_TYPE
    ts_category = 'COLOR'
    supports_format_override = False

    # Luminance (mirrors into R/G/B)
    in_low_l:  _all_props['in_low_l']
    in_mid_l:  _all_props['in_mid_l']
    in_high_l: _all_props['in_high_l']
    out_low_l: _all_props['out_low_l']
    out_high_l: _all_props['out_high_l']

    # Red
    in_low_r:  _all_props['in_low_r']
    in_mid_r:  _all_props['in_mid_r']
    in_high_r: _all_props['in_high_r']
    out_low_r: _all_props['out_low_r']
    out_high_r: _all_props['out_high_r']

    # Green
    in_low_g:  _all_props['in_low_g']
    in_mid_g:  _all_props['in_mid_g']
    in_high_g: _all_props['in_high_g']
    out_low_g: _all_props['out_low_g']
    out_high_g: _all_props['out_high_g']

    # Blue
    in_low_b:  _all_props['in_low_b']
    in_mid_b:  _all_props['in_mid_b']
    in_high_b: _all_props['in_high_b']
    out_low_b: _all_props['out_low_b']
    out_high_b: _all_props['out_high_b']

    # Alpha
    in_low_a:  _all_props['in_low_a']
    in_mid_a:  _all_props['in_mid_a']
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
        for field in _FIELDS:
            layout.prop(self, f"{field}_{ch}")

    def get_parameters(self):
        return [
            float(getattr(self, f"{field}_{ch}"))
            for ch in ('l', 'r', 'g', 'b', 'a')
            for field in _FIELDS
        ] + [float(_CHANNEL_INDEX[self.channel_mode])]

    def get_named_parameters(self):
        return None


NODE_CLASS = TS_Levels_Node
