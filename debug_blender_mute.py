"""
Debug script to run in Blender MCP.
Reproduces: simplex → invert(muted) → separate_rgba(active)
Traces connection remapping and graph building.
"""

import bpy
import sys

# Import the addon modules
addon_path = r"C:\Users\User\AppData\Roaming\Blender Foundation\Blender\5.0\extensions\user_default\texturesynth"
if addon_path not in sys.path:
    sys.path.insert(0, addon_path)

from core.engine_bridge import _build_graph_and_params, _get_active_subgraph_topo_order, _build_id_maps

# Create a test scene with the exact node setup
bpy.ops.object.shader_nodes_tree_new()
tree = bpy.context.scene.world.use_nodes = True
world_tree = bpy.data.worlds["World"].node_tree

# Clear default nodes
world_tree.nodes.clear()

# Add nodes
simplex_node = world_tree.nodes.new("TS_Simplex_Node")
simplex_node.name = "Simplex"

invert_node = world_tree.nodes.new("TS_Invert_Node")
invert_node.name = "Invert"
invert_node.mute = True  # MUTED!

sep_rgba_node = world_tree.nodes.new("TS_Separate_RGBA_Node")
sep_rgba_node.name = "SeparateRGBA"

output_node = world_tree.nodes.new("TS_Output_Node")
output_node.name = "Output"

# Create links: simplex → invert → separate_rgba
link1 = world_tree.links.new(simplex_node.outputs[0], invert_node.inputs[0])
link2 = world_tree.links.new(invert_node.outputs[0], sep_rgba_node.inputs[0])
link3 = world_tree.links.new(sep_rgba_node.outputs[0], output_node.inputs[0])

# Set separate_rgba as active node
world_tree.nodes.active = sep_rgba_node

print("\n" + "="*70)
print("BLENDER SETUP TEST: simplex → invert(muted) → separate_rgba(active)")
print("="*70)
print(f"\nNodes in tree:")
for n in world_tree.nodes:
    print(f"  {n.name:20} | muted={getattr(n, 'mute', False):5} | type={n.bl_idname}")

print(f"\nLinks in tree:")
for link in world_tree.links:
    print(f"  {link.from_node.name}[{list(link.from_node.outputs).index(link.from_socket)}] → {link.to_node.name}[{list(link.to_node.inputs).index(link.to_socket)}]")

# Build ID maps
name_to_id, id_to_name = _build_id_maps(world_tree)
print(f"\nID map:")
for name, nid in name_to_id.items():
    print(f"  {name:20} → {nid}")

# Get active subgraph order
print(f"\nCalling _get_active_subgraph_topo_order()...")
order = _get_active_subgraph_topo_order(world_tree, output_node, name_to_id)
reachable_ids = {name_to_id[n] for n in [n.name for n in world_tree.nodes] if n.name in name_to_id}

print(f"  Order: {[id_to_name.get(nid, f'unknown_{nid}') for nid in order]}")
print(f"  Reachable IDs: {sorted(reachable_ids)}")

# Now call the full graph building
print(f"\nCalling _build_graph_and_params()...")
try:
    graph, pc, node_params = _build_graph_and_params(world_tree, output_node)
    
    if graph is None:
        print("  ERROR: graph is None!")
        print("  This means output_src_id could not be resolved.")
    else:
        print("  Graph built successfully!")
        print(f"  Output source ID is set in C++")
        
except Exception as e:
    print(f"  EXCEPTION: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("END TEST")
print("="*70)
