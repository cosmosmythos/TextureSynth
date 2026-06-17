"""Manual operator to force rebuild textures from the current graph"""
import bpy

class TEXTURESYNTH_OT_update(bpy.types.Operator):
    bl_idname = "texturesynth.update"
    bl_label = "Update"
    bl_description = "Rebuild textures from the current graph"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        from ..core.evaluation import request_topology_update
        request_topology_update()
        self.report({'INFO'}, "Graph compiled")
        return {'FINISHED'}

classes = (
    TEXTURESYNTH_OT_update,
)

register, unregister = bpy.utils.register_classes_factory(classes)
