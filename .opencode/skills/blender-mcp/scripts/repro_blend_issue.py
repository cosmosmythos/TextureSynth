"""Build a clean blur→blend test graph in Blender and debug the fusion."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mcp_fetch import send

code = r"""
import bpy, sys, time, struct, json

result = {"steps": []}

# Enable C++ logging
try:
    import texturesynth_core as tsc
    def log_cb(level, msg):
        pass  # just capturing to C++ side
    tsc.set_log_callback(log_cb)
    result["steps"].append("log_callback_set")
except Exception as ex:
    result["steps"].append(f"log_failed: {ex}")

# Find the TS tree
tree = None
for t in bpy.data.node_groups:
    if t.bl_idname == "TextureSynthTreeType":
        tree = t
        break
if tree is None:
    result["error"] = "No TextureSynth tree"
else:
    # Remember current nodes
    result["existing_nodes"] = [n.name for n in tree.nodes]
    result["existing_links"] = len(tree.links)
    
    # The existing graph already has: Perlin.001 → Levels → Blur.002 → Blend (A), Invert.002 → Blend (B)
    # Blend.mask = 1.0 means output = A = Blur.002 output
    # When Blur.002 is active → 2D blur works
    # When Blend is active → only horizontal blur
    # Let's toggle active node and check output
    
    # Get engine
    from bl_ext.user_default.texturesynth.core.cpp_module import get_engine
    e = get_engine()
    if e is None or not e.has_pipeline():
        result["error"] = "Engine not ready"
    else:
        result["engine_ok"] = True
        
        # Read current params
        result["params_before"] = {}
        try:
            result["params_before"]["param_layout"] = e.param_layout()
        except:
            pass
        try:
            result["params_before"]["current_graph_output"] = e.current_graph().output_node
        except:
            pass
        
        # Force re-render by toggling active to Blend then back
        # Actually let's set Blend as active and examine
        for n in tree.nodes:
            if n.name == "Blend":
                tree.nodes.active = n
                n.select = True
            else:
                n.select = False
        
        # Force the addon to re-evaluate
        # The timer will pick up the active node change
        result["steps"].append("set_Blend_active")
        result["active_node"] = "Blend"
        
        # Try to read back result
        try:
            pc = tsc.PushConstants()
            pc.resolution_x = 256
            pc.resolution_y = 256
            pc.seed = 1
            pc.time = 0.0
            
            gen = e.current_revision()
            e.submit_render(pc, gen)
            import time
            time.sleep(0.5)
            arr = e.readback_sync()
            result["readback_shape"] = list(arr.shape)
            result["readback_sample"] = [round(float(arr[128,128,c]), 6) for c in range(4)]
        except Exception as ex:
            result["readback_error"] = str(ex)

import json as _json
print(_json.dumps(result, indent=2, default=str))
"""

resp = send(code)
import json
print(json.dumps(resp, indent=2, default=str))
