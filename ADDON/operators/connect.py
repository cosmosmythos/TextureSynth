"""Connector operators for node tree shortcuts (Blend insertion and Output auto-wire)."""
import bpy
from mathutils import Vector
from ..utils.node_utils import offset_node_location, frame_adjust
from ..utils.rna import register_class, unregister_class


# -- Helpers

def _ts_tree(context):
    sd = context.space_data
    if sd and sd.type == 'NODE_EDITOR' and sd.tree_type == 'TextureSynthTreeType':
        return sd.edit_tree
    return None

def _node_under_cursor(tree, mouse_region, region):
    """Return the topmost node at the region-space mouse position."""
    x, y = region.view2d.region_to_view(mouse_region[0], mouse_region[1])
    hit = None
    for n in tree.nodes:
        if n.bl_idname == 'NodeFrame':
            continue
        loc = n.location
        w, h = n.dimensions
        if loc.x <= x <= loc.x + w and loc.y - h <= y <= loc.y:
            hit = n
    return hit

def _first_output(node):
    return next((s for s in node.outputs if s.enabled), None)

def _first_free_input(node):
    return next((s for s in node.inputs if s.enabled and not s.is_linked), None)


# -- Blend Connector

class TS_OT_connect_blend(bpy.types.Operator):
    bl_idname  = "texturesynth.connect_blend"
    bl_label   = "Connect with Blend"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _ts_tree(context) is not None

    def invoke(self, context, event):
        self._tree = _ts_tree(context)
        self._region = context.region
        self._src = _node_under_cursor(
            self._tree, (event.mouse_region_x, event.mouse_region_y), self._region)
        if self._src is None:
            return {'CANCELLED'}
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'MOUSEMOVE':
            return {'RUNNING_MODAL'}
        if event.type == 'RIGHTMOUSE' and event.value == 'RELEASE':
            dst = _node_under_cursor(
                self._tree, (event.mouse_region_x, event.mouse_region_y), self._region)
            if dst is None or dst == self._src:
                return {'CANCELLED'}
            self._insert_blend(self._src, dst)
            return {'FINISHED'}
        if event.type in {'ESC', 'LEFTMOUSE'}:
            return {'CANCELLED'}
        return {'RUNNING_MODAL'}

    def _insert_blend(self, a, b):
        t = self._tree
        oa = _first_output(a)
        if oa is None:
            self.report({'WARNING'}, "Source node has no output.")
            return
        blend = t.nodes.new('TS_Blend_Node')
        blend.location = (a.location + b.location) * 0.5 + Vector((180, 0))

        # Check if a is already wired to b (INSERT mode).
        existing_link = None
        to_socket = None
        for link in list(t.links):
            if link.from_node == a and link.to_node == b:
                existing_link = link
                to_socket = link.to_socket
                break

        if existing_link and to_socket:
            # INSERT: a → blend.A → b (break the original link)
            # Blend socket order: [0]=Mask, [1]=A, [2]=B.
            t.links.remove(existing_link)
            t.links.new(oa, blend.inputs[1])
            t.links.new(blend.outputs[0], to_socket)
        else:
            # MERGE: a → blend.A, b → blend.B, blend → Output
            t.links.new(oa, blend.inputs[1])
            ob = _first_output(b)
            if ob:
                t.links.new(ob, blend.inputs[2])
            out = next((n for n in t.nodes if n.bl_idname == 'TS_Output_Node'), None)
            if out:
                isock = out.inputs[0]
                if isock.is_linked:
                    for l in list(isock.links):
                        t.links.remove(l)
                t.links.new(blend.outputs[0], isock)


# -- Output Connector

class TS_OT_connect_to_output(bpy.types.Operator):
    bl_idname  = "texturesynth.connect_to_output"
    bl_label   = "Connect to Output"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _ts_tree(context) is not None

    def invoke(self, context, event):
        tree = _ts_tree(context)
        src = _node_under_cursor(
            tree, (event.mouse_region_x, event.mouse_region_y), context.region)
        if src is None or src.bl_idname == 'TS_Output_Node':
            return {'CANCELLED'}
        out = next((n for n in tree.nodes if n.bl_idname == 'TS_Output_Node'), None)
        if out is None:
            out = tree.nodes.new('TS_Output_Node')
            offset_node_location(src, out, [100, 0])
            frame_adjust(src, out)
        osock = _first_output(src)
        isock = out.inputs[0]
        if osock is None:
            return {'CANCELLED'}
        for l in list(isock.links):
            tree.links.remove(l)
        tree.links.new(osock, isock)
        return {'FINISHED'}


# -- Keymaps

_KEYMAPS = []

def _register_keymaps():
    kc = bpy.context.window_manager.keyconfigs.addon
    if not kc:
        return
    km = kc.keymaps.new(name="Node Editor", space_type='NODE_EDITOR')

    kmi = km.keymap_items.new(
        TS_OT_connect_blend.bl_idname, 'RIGHTMOUSE', 'PRESS',
        ctrl=True, alt=True)
    _KEYMAPS.append((km, kmi))

    kmi = km.keymap_items.new(
        TS_OT_connect_to_output.bl_idname, 'LEFTMOUSE', 'RELEASE',
        ctrl=True, alt=True)
    _KEYMAPS.append((km, kmi))


classes = (TS_OT_connect_blend, TS_OT_connect_to_output)

def register():
    for c in classes:
        register_class(c)
    _register_keymaps()

def unregister():
    for km, kmi in _KEYMAPS:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _KEYMAPS.clear()
    for c in reversed(classes):
        unregister_class(c)