"""Output node"""

from ..base import TextureSynthNode


SV_TYPE = None   # UI-only, no JSON manifest, not in skip-set


class TS_Output_Node(TextureSynthNode):
    bl_idname = 'TS_Output_Node'
    bl_label  = 'Output'
    sv_type   = None   # consumed by engine_bridge to identify the sink
    ts_category = 'OUTPUT'

    def init(self, context):
        super().init(context)
        self.inputs.new('TextureSynthSocketType', "Result")


NODE_CLASS = TS_Output_Node