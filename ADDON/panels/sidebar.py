"""
Sidebar panel for TextureSynth — Node Editor N-panel.

Shows engine status, resolution control, and manual update button.
"""

import bpy
from ..core import cpp_module


class TEXTURESYNTH_PT_sidebar(bpy.types.Panel):
    bl_label = "TextureSynth"
    bl_idname = "TEXTURESYNTH_PT_sidebar"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "TextureSynth"

    @classmethod
    def poll(cls, context):
        return (
            context.space_data is not None
            and hasattr(context.space_data, 'tree_type')
            and context.space_data.tree_type == 'TextureSynthTreeType'
        )

    def draw(self, context):
        layout = self.layout

        # Engine status
        box = layout.box()
        if cpp_module.is_loaded():
            box.label(text="Engine: Ready", icon='CHECKMARK')
        else:
            box.label(text="Engine: Not loaded", icon='ERROR')

        # Resolution and format controls
        col = layout.column(align=True)
        col.prop(context.scene, "texturesynth_resolution")
        col.prop(context.scene, "texturesynth_precision")
        col.prop(context.scene, "texturesynth_proxy_scale")

        # Manual update
        layout.operator("texturesynth.update", icon='FILE_REFRESH')


classes = (
    TEXTURESYNTH_PT_sidebar,
)


def on_precision_update(self, context):
    from ..core.evaluation import request_topology_update
    request_topology_update()


def on_proxy_scale_update(self, context):
    from ..core.evaluation import request_param_update
    request_param_update()


def register():
    bpy.types.Scene.texturesynth_resolution = bpy.props.IntProperty(
        name="Resolution",
        description="Output texture resolution (square)",
        default=512,
        min=64,
        max=4096,
        subtype='PIXEL',
    )
    bpy.types.Scene.texturesynth_precision = bpy.props.EnumProperty(
        name="Precision",
        description="VRAM Bit-Depth precision",
        items=[
            ('R32', "32-bit Float (High Quality)", "Use 32-bit single-precision floating point"),
            ('R16', "16-bit Half-Float (Optimized)", "Use 16-bit half-precision floating point (saves 50% VRAM)"),
            ('R8', "8-bit Int (Preview)", "Use 8-bit integer formats (saves 75% VRAM)"),
        ],
        default='R16',
        update=on_precision_update,
    )
    bpy.types.Scene.texturesynth_proxy_scale = bpy.props.EnumProperty(
        name="Viewport Proxy",
        description="Copernicus-style viewport proxy rendering scale",
        items=[
            ('1.0', "100% (Full Quality)", "Render at full resolution"),
            ('0.5', "50% (Fast)", "Render at half resolution and upscale"),
            ('0.25', "25% (Extremely Fast)", "Render at quarter resolution and upscale"),
        ],
        default='1.0',
        update=on_proxy_scale_update,
    )
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "texturesynth_resolution"):
        del bpy.types.Scene.texturesynth_resolution
    if hasattr(bpy.types.Scene, "texturesynth_precision"):
        del bpy.types.Scene.texturesynth_precision
    if hasattr(bpy.types.Scene, "texturesynth_proxy_scale"):
        del bpy.types.Scene.texturesynth_proxy_scale
