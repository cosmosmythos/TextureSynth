"""Texture-tree-aware connector ops.
   • Ctrl+Shift+RMB drag from node A onto node B -> insert Blend(A,B) between them.
   • Ctrl+Shift+LMB release on a node           -> wire its first output to Output node.
     (Uses RELEASE event instead of CLICK to avoid interference with node selection)
"""
import bpy
from mathutils import Vector
from ..utils.node_utils import offset_node_location, frame_adjust


# ── helpers ──────────────────────────────────────────────────────────
def _ts_tree(context):
    sd = context.space_data
    if sd and sd.type == 'NODE_EDITOR' and sd.tree_type == 'TextureSynthTreeType':
        return sd.edit_tree
    return None

def _node_under_cursor(tree, mouse_region, region):
    """Return the topmost node whose rect contains the region-space mouse pos."""
    # node.location is in tree space; we need to convert region pixels → tree space.
    # Use region_to_view to get tree-space coords.
    x, y = region.view2d.region_to_view(mouse_region[0], mouse_region[1])
    hit = None
    for n in tree.nodes:
        if n.bl_idname == 'NodeFrame':
            continue
        loc = n.location
        w, h = n.dimensions
        if loc.x <= x <= loc.x + w and loc.y - h <= y <= loc.y:
            hit = n  # last hit = topmost in iter order
    return hit

def _first_output(node):
    return next((s for s in node.outputs if s.enabled), None)

def _first_free_input(node):
    return next((s for s in node.inputs if s.enabled and not s.is_linked), None)


# ── Ctrl+Shift+RMB drag: insert Blend(A, B) ──────────────────────────
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
        oa, ob = _first_output(a), _first_output(b)
        if oa is None or ob is None:
            self.report({'WARNING'}, "Source nodes need outputs.")
            return
        blend = t.nodes.new('TS_Blend_Node')
        blend.location = (a.location + b.location) * 0.5 + Vector((180, 0))
        t.links.new(oa, blend.inputs[0])
        t.links.new(ob, blend.inputs[1])
        # Optional: auto-wire to existing Output if present and free.
        out = next((n for n in t.nodes if n.bl_idname == 'TS_Output_Node'), None)
        if out and not out.inputs[0].is_linked:
            t.links.new(blend.outputs[0], out.inputs[0])


# ── Ctrl+Shift+LMB click: wire node → Output ─────────────────────────
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
            # Position output node to the right of source node, using Sverchok pattern
            offset_node_location(src, out, [100, 0])
            # If source node is in a frame, place output in same frame
            frame_adjust(src, out)
        osock = _first_output(src)
        isock = out.inputs[0]
        if osock is None:
            return {'CANCELLED'}
        # Replace any existing link.
        for l in list(isock.links):
            tree.links.remove(l)
        tree.links.new(osock, isock)
        return {'FINISHED'}


# ── keymap registration ─────────────────────────────────────────────
_KEYMAPS = []

def _register_keymaps():
    kc = bpy.context.window_manager.keyconfigs.addon
    if not kc:
        return
    km = kc.keymaps.new(name="Node Editor", space_type='NODE_EDITOR')

    # Ctrl+Shift+RMB drag
    kmi = km.keymap_items.new(
        TS_OT_connect_blend.bl_idname, 'RIGHTMOUSE', 'PRESS',
        ctrl=True, shift=True)
    _KEYMAPS.append((km, kmi))

    # Ctrl+Shift+LMB release (Sverchok pattern: RELEASE avoids node-selection interference)
    kmi = km.keymap_items.new(
        TS_OT_connect_to_output.bl_idname, 'LEFTMOUSE', 'RELEASE',
        ctrl=True, shift=True)
    _KEYMAPS.append((km, kmi))


classes = (TS_OT_connect_blend, TS_OT_connect_to_output)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    _register_keymaps()

def unregister():
    for km, kmi in _KEYMAPS:
        try: km.keymap_items.remove(kmi)
        except Exception: pass
    _KEYMAPS.clear()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)