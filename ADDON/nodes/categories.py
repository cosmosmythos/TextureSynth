import bpy
from ..utils.rna import register_class, unregister_class

_CATEGORY_ORDER = ('NOISE', 'INPUT', 'FILTER', 'BLEND', 'COLOR', 'CHANNEL', 'OUTPUT')

_CATEGORY_LABELS = {
    'NOISE':  'Generators',
    'INPUT':  'Input',
    'FILTER': 'Filters',
    'BLEND':  'Blend',
    'COLOR':  'Color',
    'CHANNEL': 'Channel',
    'OUTPUT': 'Output',
}


def _build_category_menus():
    """Create one NODE_MT_sub_texturesynth_<cat> submenu per category."""
    menus = {}
    for cat in _CATEGORY_ORDER:
        cls_name = f"NODE_MT_sub_texturesynth_{cat.lower()}"
        label = _CATEGORY_LABELS.get(cat, cat.title())

        def _make_draw(category):
            def draw(self, context):
                from . import factory
                for cls in factory.get_generated_classes():
                    if getattr(cls, 'ts_category', 'FILTER') == category:
                        op = self.layout.operator("node.add_node", text=cls.bl_label)
                        op.type = cls.bl_idname
                        op.use_transform = True
            return draw

        menu_cls = type(cls_name, (bpy.types.Menu,), {
            'bl_idname': cls_name,
            'bl_label': label,
            'draw': _make_draw(cat),
        })
        menus[cat] = menu_cls
    return menus


_category_menus = _build_category_menus()


def _draw_add_menu(self, context):
    if context.space_data.tree_type != 'TextureSynthTreeType':
        return
    layout = self.layout
    layout.separator()
    for cat in _CATEGORY_ORDER:
        cls_name = f"NODE_MT_sub_texturesynth_{cat.lower()}"
        label = _CATEGORY_LABELS.get(cat, cat.title())
        layout.menu(cls_name, text=label)

classes = tuple(_category_menus.values())


def _register_extra():
    bpy.types.NODE_MT_add.append(_draw_add_menu)


def _unregister_extra():
    bpy.types.NODE_MT_add.remove(_draw_add_menu)


# Wrap register/unregister to add/remove the draw callback on NODE_MT_add.
def register():
    for cls in classes:
        register_class(cls)
    _register_extra()


def unregister():
    _unregister_extra()
    for cls in reversed(classes):
        unregister_class(cls)
