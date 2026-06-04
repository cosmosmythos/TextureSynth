"""Output node — multi-input sink with named targets.

The output node holds a CollectionProperty of `TS_OutputTarget` rows
(name + source-node-id). On Bake, the engine iterates these targets in
order and produces one bpy.data.images.Image per target. The
`TS_Output_Node.targets` list is the source of truth for what to bake.

Live preview follows the link feeding the 'Result' input (preserves the
existing single-output preview behavior); targets only matter on Bake.
"""
import bpy
from bpy.types import Node
from bpy.props import (
    StringProperty, IntProperty,
    CollectionProperty, PointerProperty,
)
from ..base import TextureSynthNode


class TS_OutputTarget(bpy.types.PropertyGroup):
    """One row in the output node's 'targets' list. Each target becomes
    a separate bpy.data.images.Image on Bake. The name is also used as
    the image name (uniqueness enforced by Blender's data-block system).
    """
    name: StringProperty(
        name="Name",
        default="Base Color",
        description="Texture name. Becomes the bpy image name on Bake.",
    )
    # Source node identity: the stable_id (uint64, derived from the source
    # node's UUID — see TextureSynthNode.stable_id()). We store the full
    # hex string because bpy's IntProperty is signed int32 and silently
    # truncates uint64. The bake operator resolves this string to an int
    # at bake time.
    source_node: StringProperty(
        name="Source Node ID",
        default="",
        description="Stable ID of the graph node whose output this target renders. "
                    "Right-click a node -> 'Copy stable ID', or leave blank to "
                    "auto-resolve from the Output node's 'Result' input link.",
    )


class TS_Output_Node(TextureSynthNode):
    """The 'bake sink'. Has a 'Result' input for the live preview link
    and an ordered list of named targets (Base Color / Normal / Height /
    ...) for the Bake button in the N-panel.
    """
    bl_idname = 'TS_Output_Node'
    bl_label  = 'Output'
    sv_type   = None   # consumed by engine_bridge to identify the sink
    ts_category = 'OUTPUT'

    targets: CollectionProperty(type=TS_OutputTarget)
    active_target_index: IntProperty(
        name="Active Target Index",
        default=0,
    )

    def init(self, context):
        super().init(context)
        self.inputs.new('TS_DefaultSocketType', "Result")
        # Seed with the conventional PBR triad: Base Color / Normal / Roughness.
        # The user can rename, remove, or add more with the buttons in
        # draw_buttons below.
        for default_name in ("Base Color", "Normal", "Roughness"):
            t = self.targets.add()
            t.name = default_name

    def draw_buttons(self, context, layout):
        col = layout.column(align=True)
        # Header row: + and - buttons
        row = col.row(align=True)
        op = row.operator("texturesynth.output_target_add", text="", icon="ADD")
        op = row.operator("texturesynth.output_target_remove", text="", icon="REMOVE")
        col.separator()
        if not self.targets:
            col.label(text="No targets — click + to add", icon="INFO")
            return
        # Show one row per target: name on the left, source id on the right.
        # The active row is highlighted (depress).
        idx = max(0, min(self.active_target_index, len(self.targets) - 1))
        for i, t in enumerate(self.targets):
            row = col.row(align=True)
            sel = row.operator("texturesynth.output_target_select",
                               text="",
                               icon="LAYER_ACTIVE" if i == idx else "LAYER_USED",
                               depress=(i == idx))
            sel.index = i
            row.prop(t, "name", text="")
            row.prop(t, "source_node", text="Src")


NODE_CLASS = TS_Output_Node
