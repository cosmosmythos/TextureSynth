import bpy

class NODE_MT_category_texturesynth(bpy.types.Menu):
    bl_idname = "NODE_MT_category_texturesynth"
    bl_label = "TextureSynth"

    def draw(self, context):
        layout = self.layout
        
        def add_menu_item(text, node_type):
            op = layout.operator("node.add_node", text=text)
            op.type = node_type
            op.use_transform = True
            
        from . import factory
        for cls in factory.get_generated_classes():
            add_menu_item(cls.bl_label, cls.bl_idname)

def _draw_add_menu(self, context):
    if context.space_data.tree_type != 'TextureSynthTreeType':
        return
    layout = self.layout
    layout.separator()
    layout.menu("NODE_MT_category_texturesynth", text="TextureSynth")

classes = (
    NODE_MT_category_texturesynth,
)


def _register_extra():
    bpy.types.NODE_MT_add.append(_draw_add_menu)


def _unregister_extra():
    bpy.types.NODE_MT_add.remove(_draw_add_menu)


# Wrap register/unregister to add/remove draw callback to NODE_MT_add.
def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    _register_extra()


def unregister():
    _unregister_extra()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
