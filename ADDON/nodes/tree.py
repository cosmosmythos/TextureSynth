"""
Node Tree and Socket definitions for TextureSynth.

Pure UI — no engine logic, no imports from core/.
"""

import bpy


class TextureSynthSocket(bpy.types.NodeSocket):
    """Custom socket type for passing texture data between nodes."""
    bl_idname = 'TextureSynthSocketType'
    bl_label = "Texture Socket"

    def draw(self, context, layout, node, text):
        if self.is_linked:
            layout.label(text=text)
        elif node and hasattr(node, self.name):
            layout.prop(node, self.name, text=text)
        else:
            layout.label(text=text)

    @classmethod
    def draw_color_simple(cls):
        return (0.78, 0.78, 0.2, 1.0)


class TextureSynthTree(bpy.types.NodeTree):
    """Custom node tree for procedural texture authoring."""
    bl_idname = 'TextureSynthTreeType'
    bl_label = "TextureSynth Editor"
    bl_icon = 'TEXTURE'

    @classmethod
    def poll(cls, context):
        return True

    def update(self):
        """Called when connections are made/broken (topology change)."""
        try:
            from ..core.evaluation import request_topology_update
            request_topology_update()
        except Exception:
            pass


classes = (
    TextureSynthSocket,
    TextureSynthTree,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
