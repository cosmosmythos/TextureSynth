"""Repeat zone — single add operator that spawns Begin + End pair in a colored frame."""
import bpy
from ..base import TextureSynthNode

# Orange tint matching Blender's native zone look.
_ZONE_COLOR = (0.8, 0.4, 0.1)


class TS_RepeatBegin_Node(TextureSynthNode):
    bl_idname = 'TS_RepeatBegin_Node'
    bl_label = 'Repeat'
    sv_type = None
    ts_category = 'HIDDEN'

    iterations: bpy.props.IntProperty(
        name="Iterations",
        description="Number of times to loop",
        default=3,
        min=1,
        max=100,
        update=lambda self, ctx: _request_update(),
    )

    def init(self, context):
        super().init(context)
        self.inputs.new('TS_DefaultSocketType', "Initial")
        self.outputs.new('TS_DefaultSocketType', "Current")

    def draw_buttons(self, context, layout):
        layout.prop(self, "iterations")


class TS_RepeatEnd_Node(TextureSynthNode):
    bl_idname = 'TS_RepeatEnd_Node'
    bl_label = 'Repeat'
    sv_type = None
    ts_category = 'HIDDEN'

    def init(self, context):
        super().init(context)
        self.inputs.new('TS_DefaultSocketType', "Result")
        self.outputs.new('TS_DefaultSocketType', "Final")


class TS_OT_repeat_add(bpy.types.Operator):
    """Add a repeat zone — two paired nodes with a colored frame"""
    bl_idname = "texturesynth.repeat_add"
    bl_label = "Repeat"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        sd = context.space_data
        return (sd and sd.type == 'NODE_EDITOR'
                and sd.tree_type == 'TextureSynthTreeType'
                and sd.edit_tree is not None)

    def execute(self, context):
        tree = context.space_data.edit_tree
        nodes = tree.nodes
        links = tree.links

        begin = nodes.new('TS_RepeatBegin_Node')
        begin.location = (0, 0)
        end = nodes.new('TS_RepeatEnd_Node')
        end.location = (400, 0)

        frame = nodes.new('NodeFrame')
        frame.label = "Repeat"
        frame.shrink = True
        frame.use_custom_color = True
        frame.color = _ZONE_COLOR
        begin.parent = frame
        end.parent = frame

        # No visible feedback wire — the engine loops End→Begin internally.
        # Users wire: Begin.Current → [body nodes] → End.Result

        _fit_frame(frame, [begin, end], padding=40)

        for n in nodes:
            n.select = False
        begin.select = True
        end.select = True

        return {'FINISHED'}


def _fit_frame(frame, child_nodes, padding=40):
    if not child_nodes:
        return
    min_x = min(n.location.x for n in child_nodes)
    max_x = max(n.location.x + n.dimensions.x for n in child_nodes)
    min_y = min(n.location.y - n.dimensions.y for n in child_nodes)
    max_y = max(n.location.y for n in child_nodes)
    frame.location = (min_x - padding, max_y + padding)


def _request_update():
    try:
        from ...core.evaluation import request_param_update
        request_param_update()
    except Exception:
        pass


NODE_CLASSES = (TS_RepeatBegin_Node, TS_RepeatEnd_Node)
OPERATOR_CLASSES = (TS_OT_repeat_add,)
SOCKET_CLASSES = ()
PROPERTY_GROUPS = ()
