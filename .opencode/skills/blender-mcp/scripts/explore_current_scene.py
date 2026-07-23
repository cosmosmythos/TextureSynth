"""Get full scene graph: nodes, links, params."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mcp_fetch import send

code = r"""
import bpy, sys
result = {"nodes": [], "links": []}
tree = None
for t in bpy.data.node_groups:
    if t.bl_idname == "TextureSynthTreeType":
        tree = t
        break
if tree is None:
    result["error"] = "No TextureSynth tree"
else:
    result["tree_name"] = tree.name
    for n in tree.nodes:
        info = {"name": n.name, "type": n.bl_rna.identifier, "location": [round(n.location.x,1), round(n.location.y,1)], "select": n.select, "mute": n.mute}
        props = {}
        for p in n.bl_rna.properties:
            if p.identifier in ("rna_type","name","location","select","mute","type","inputs","dimensions","width","height","label","parent","panel_states","internal_links","warning_propagation","use_custom_color","color","color_tag","show_options","show_preview","hide","show_texture","bl_idname","bl_label","bl_description","bl_icon","bl_static_type","bl_width_default","bl_width_min","bl_width_max","bl_height_default","bl_height_min","bl_height_max","ts_uuid","ts_compile_error","format_override","depth_mode","absolute_depth","location_absolute"):
                continue
            try:
                val = getattr(n, p.identifier)
                if hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
                    val = [round(float(v), 4) for v in val]
                props[p.identifier] = val
            except:
                pass
        info["props"] = props
        result["nodes"].append(info)
    for l in tree.links:
        result["links"].append({"from": (l.from_node.name, l.from_socket.name if l.from_socket else "?"), "to": (l.to_node.name, l.to_socket.name if l.to_socket else "?")})
    if tree.nodes.active:
        an = tree.nodes.active
        result["active_node"] = {"name": an.name, "type": an.bl_rna.identifier}
try:
    from bl_ext.user_default.texturesynth.core.cpp_module import get_engine
    e = get_engine()
    if e is not None and e.has_pipeline():
        result["output_node"] = e.current_graph().output_node
except:
    pass
"""

resp = send(code)
import json
print(json.dumps(resp, indent=2, default=str))
