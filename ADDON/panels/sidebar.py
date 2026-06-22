"""Sidebar panel."""
import bpy
from ..core import cpp_module


class TEXTURESYNTH_PT_sidebar(bpy.types.Panel):
    bl_label = "Settings"
    bl_idname = "TEXTURESYNTH_PT_sidebar"
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Options"

    @classmethod
    def poll(cls, context):
        return (
            context.space_data is not None
            and hasattr(context.space_data, 'tree_type')
            and context.space_data.tree_type == 'TextureSynthTreeType'
        )

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        if cpp_module.is_loaded():
            box.label(text="GPU: Active", icon='TEXTURE')
        else:
            box.label(text="GPU: Unavailable", icon='NOT_FOUND')

        col = layout.column(align=True)
        col.prop(context.scene, "texturesynth_resolution")
        col.prop(context.scene, "texturesynth_precision", text="")
        col.prop(context.scene, "texturesynth_proxy_scale", text="")

        layout.operator("texturesynth.update", icon='FILE_REFRESH')


classes = (
    TEXTURESYNTH_PT_sidebar,
)


def on_precision_update(self, context):
    from ..core.evaluation import request_topology_update
    request_topology_update()


def on_proxy_scale_update(self, context):
    # Render resolution changed, trigger topology update to resize allocations.
    from ..core.evaluation import request_topology_update
    request_topology_update()


def _register_extra():
    """Add Scene custom properties used by the panel."""
    bpy.types.Scene.texturesynth_resolution = bpy.props.IntProperty(
        name="Resolution",
        description="Resolution",
        default=1024,
        min=64,
        max=4096,
        subtype='PIXEL',
    )
    bpy.types.Scene.texturesynth_precision = bpy.props.EnumProperty(
        name="Default Precision",
        description="Graph-level default bit depth. Per-node 'Depth' overrides take precedence (Substance Designer-style inheritance)",
        items=[
            ('R32', "32-bit Float", ""),
            ('R16', "16-bit Half-Float", ""),
            ('R8', "8-bit", ""),
        ],
        default='R16',
        update=on_precision_update,
    )
    bpy.types.Scene.texturesynth_proxy_scale = bpy.props.EnumProperty(
        name="Proxy Scale",
        description="",
        items=[
            ('1.0', "100%", ""),
            ('0.5', "50%", ""),
            ('0.25', "25%", ""),
        ],
        default='1.0',
        update=on_proxy_scale_update,
    )


def _unregister_extra():
    if hasattr(bpy.types.Scene, "texturesynth_resolution"):
        del bpy.types.Scene.texturesynth_resolution
    if hasattr(bpy.types.Scene, "texturesynth_precision"):
        del bpy.types.Scene.texturesynth_precision
    if hasattr(bpy.types.Scene, "texturesynth_proxy_scale"):
        del bpy.types.Scene.texturesynth_proxy_scale


# Wrap registration to also register/unregister Scene properties.
def register():
    _register_extra()
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    _unregister_extra()
