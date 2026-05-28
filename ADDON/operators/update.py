"""
Manual "Update Texture" operator.

Useful when auto-update is off, or for debugging.
"""

import bpy


class TEXTURESYNTH_OT_update(bpy.types.Operator):
    bl_idname = "texturesynth.update"
    bl_label = "Update Texture"
    bl_description = "Force a texture recompute from the current node graph"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        from ..core.evaluation import request_topology_update
        request_topology_update()
        self.report({'INFO'}, "TextureSynth update requested.")
        return {'FINISHED'}


classes = (
    TEXTURESYNTH_OT_update,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
