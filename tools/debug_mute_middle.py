#!/usr/bin/env python3
"""
Debug script to test the mute middle-node case:
simplex → invert(muted) → blend(active)
"""

import sys
sys.path.insert(0, r'C:\Users\User\Documents\0\TEXTURESYNTH\ADDON\core')

from engine_bridge import _build_graph_and_params, _node_stable_id
import texturesynth as cpp_module

# Simulate Blender tree structure
class FakeLink:
    def __init__(self, from_node, from_socket_idx, to_node, to_socket_idx):
        self.from_node = from_node
        self.to_node = to_node
        self.from_socket = FakeSocket("Output", from_socket_idx)
        self.to_socket = FakeSocket("Input", to_socket_idx)
        self.is_valid = True
        self.links = [self]

class FakeSocket:
    def __init__(self, name, idx):
        self.name = name
        self.index = idx
        self.is_linked = True

class FakeNode:
    def __init__(self, name, node_type, mute=False):
        self.name = name
        self.bl_idname = f"TS_{node_type}_Node"
        self.sv_type = node_type
        self.mute = mute
        self.outputs = [FakeSocket("Output", 0)]
        self.inputs = [FakeSocket("A/In", 0), FakeSocket("B/In", 1), FakeSocket("Mask/Mask", 2)]
        self._stable_id = hash(name) & 0xFFFFFFFF

    def stable_id(self):
        return self._stable_id

    def get_format_override(self):
        return None

class FakeTree:
    def __init__(self):
        self.nodes = []
        self.links = []

    def add_node(self, name, node_type, mute=False):
        node = FakeNode(name, node_type, mute)
        self.nodes.append(node)
        return node

    def add_link(self, from_node, from_socket, to_node, to_socket):
        link = FakeLink(from_node, from_socket, to_node, to_socket)
        self.links.append(link)

# Create the test case
tree = FakeTree()
simplex = tree.add_node("Simplex", "Simplex")
invert = tree.add_node("Invert", "Invert", mute=True)  # MUTED
blend = tree.add_node("Blend", "Blend", mute=False)     # ACTIVE

tree.add_link(simplex, 0, invert, 0)    # simplex → invert.A
tree.add_link(invert, 0, blend, 0)      # invert → blend.A

# Set active node
class FakeNodeCollection:
    def __init__(self, active):
        self.active = active

tree.nodes = FakeNodeCollection(blend)

print("=" * 60)
print("TEST CASE: simplex → invert(muted) → blend(active)")
print("=" * 60)
print(f"Nodes: {[n.name for n in [simplex, invert, blend]]}")
print(f"Active: {blend.name}")
print(f"Links: simplex→invert, invert→blend")
print()

# Call the graph building function
try:
    graph, pc, node_params = _build_graph_and_params(tree, None)
    
    if graph is None:
        print("ERROR: _build_graph_and_params returned None!")
        print("This means output_src_id could not be resolved.")
    else:
        print("Graph built successfully!")
        print(f"Output source ID: {graph.output_src if hasattr(graph, 'output_src') else 'N/A'}")
        print()
        print("Nodes in graph:")
        # Note: We can't easily inspect the C++ graph object, so we'll note what Python sent
        print(f"  - Simplex (stable_id: {simplex.stable_id()})")
        print(f"  - Invert (stable_id: {invert.stable_id()}) [MUTED]")
        print(f"  - Blend (stable_id: {blend.stable_id()})")
        print()
        print("Links passed to C++ (all links, Python doesn't filter muted):")
        print(f"  - Simplex[0] → Invert[0]")
        print(f"  - Invert[0] → Blend[0]")
        print()
        print("Expected after C++ rewiring:")
        print(f"  IR nodes: Simplex, Blend (Invert excluded)")
        print(f"  IR connections: Simplex[0] → Blend[0]")
        print(f"  IR output_node: Blend")
        
except Exception as e:
    print(f"EXCEPTION: {e}")
    import traceback
    traceback.print_exc()
