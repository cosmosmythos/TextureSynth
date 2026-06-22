"""
Mocked-bpy unit tests for ADDON/core/engine_bridge.py preview-root resolution.

These tests isolate the PURE-PYTHON addon logic that decides which node and
which socket the engine blits to TS_Preview2D. They do NOT touch Vulkan or
the C++ engine. The motivation is the user-reported Blender symptom:

  - "Swapping blend inputs A and B produces NO change to TS_Preview2D"
  - "Multiple blend nodes: only ONE updates when active"

Both symptoms live in the addon's graph-build / output-target wiring, not in
the engine (proven by probes: a real A/B swap at the engine layer produces
|W1-W2| ~= 0.499). The param-hash short-circuit was a false suspect -- the
fingerprint guard in evaluation.py sits in front of it and submit_graph()
resets _last_pushed_param_hash=None. So the remaining suspects are:

  1. _active_subgraph_fingerprint not detecting an A/B edge swap
  2. _resolve_preview_root picking the wrong root when Output is disconnected
  3. output_src_id / add_output_target not following the active blend node

A fake Blender node tree (FakeNode/FakeSocket/FakeLink) stands in for bpy.
engine_bridge is imported under a stubbed package so `import bpy` and the
relative `from . import cpp_module` resolve to fakes. A RecordingGraph
captures the calls the real engine would receive, letting us assert on
topology + output-target wiring without a GPU.

If any of these tests FAIL, they reproduce the user's symptom at the addon
layer -- the bug is then localized and the fix is obvious. If they all PASS,
the addon logic is correct and the symptom is elsewhere (likely Blender-side
event delivery: msgbus/timer not firing on edge swap).
"""
import importlib
import sys
import types

import pytest


# --------------------------------------------------------------------------
# Fake Blender node tree
# --------------------------------------------------------------------------

class FakeSocket:
    def __init__(self, name, node, is_output=True):
        self.name = name
        self.node = node
        self.is_output = is_output
        self.is_linked = False
        self.links = []

    def __repr__(self):
        return f"<Sock {self.node.name}.{self.name}{'out' if self.is_output else 'in'}>"


class FakeLink:
    def __init__(self, from_node, from_socket, to_node, to_socket):
        self.from_node = from_node
        self.from_socket = from_socket
        self.to_node = to_node
        self.to_socket = to_socket
        self.is_valid = True


class FakeNode:
    """Minimal node proxy. sv_type set => TextureSynth node; bl_idname pickable."""
    def __init__(self, name, stable_id, sv_type=None, bl_idname='TSNode'):
        self.name = name
        self._sid = stable_id
        self.sv_type = sv_type
        self.bl_idname = bl_idname
        self.mute = False
        self.inputs = []
        self.outputs = []
        self.select = False

    def stable_id(self):
        return self._sid

    def get_parameters(self):
        return []

    def __repr__(self):
        return f"<Node {self.name} id={self._sid} sv={self.sv_type}>"


class FakeNodeCollection:
    """tree.nodes — supports iteration, .get(name), .active."""
    def __init__(self):
        self._nodes = []

    def add(self, node):
        self._nodes.append(node)
        return node

    def get(self, name):
        for n in self._nodes:
            if n.name == name:
                return n
        return None

    @property
    def active(self):
        for n in self._nodes:
            if n.select:
                return n
        return None

    def __iter__(self):
        return iter(self._nodes)


class FakeTree:
    """tree.links + tree.nodes + bl_idname."""
    def __init__(self):
        self.nodes = FakeNodeCollection()
        self.links = []
        self.bl_idname = 'TextureSynthTreeType'

    def connect(self, from_node, from_socket_idx, to_node, to_socket_idx):
        link = FakeLink(from_node, from_node.outputs[from_socket_idx],
                        to_node, to_node.inputs[to_socket_idx])
        self.links.append(link)
        link.from_socket.is_linked = True
        link.from_socket.links.append(link)
        link.to_socket.is_linked = True
        link.to_socket.links.append(link)
        return link

    def disconnect_all_to(self, to_node, to_socket_idx):
        """Remove every link feeding to_node.inputs[to_socket_idx]."""
        sock = to_node.inputs[to_socket_idx]
        keep = []
        for l in self.links:
            if l.to_socket is sock:
                l.from_socket.is_linked = False
                l.from_socket.links.remove(l)
                l.to_socket.is_linked = False
                l.to_socket.links.remove(l)
            else:
                keep.append(l)
        self.links = keep


def _make_node(tree, name, sid, sv_type, n_inputs, n_outputs,
               input_names=None, output_names=None, bl_idname='TSNode'):
    n = FakeNode(name, sid, sv_type=sv_type, bl_idname=bl_idname)
    in_names = input_names or [f"in{i}" for i in range(n_inputs)]
    out_names = output_names or [f"out{i}" for i in range(n_outputs)]
    n.inputs = [FakeSocket(in_names[i], n, is_output=False) for i in range(n_inputs)]
    n.outputs = [FakeSocket(out_names[i], n, is_output=True) for i in range(n_outputs)]
    return tree.nodes.add(n)


# --------------------------------------------------------------------------
# RecordingGraph + FakeCore (stands in for texturesynth_core)
# --------------------------------------------------------------------------

class RecordingGraph:
    def __init__(self):
        self.nodes = []           # (id, type_id, ...)
        self.connections = []     # (sid, so, did, di)
        self.output_targets = []  # (src_id, name, src_idx)
        self.output_node = None

    def add_node(self, *args, **kwargs):
        self.nodes.append((args, kwargs))

    def add_connection(self, sid, so, did, di):
        self.connections.append((sid, so, did, di))

    def add_output_target(self, src_id, name, src_idx):
        self.output_targets.append((src_id, name, src_idx))

    def set_output(self, node_id):
        self.output_node = node_id


class FakePushConstants:
    def __init__(self):
        self.resolution_x = 64
        self.resolution_y = 64
        self.seed = 1
        self.time = 0.0


class FakeCore:
    def Graph(self):
        return RecordingGraph()

    def PushConstants(self):
        return FakePushConstants()


# --------------------------------------------------------------------------
# Package fixture: install bpy + cpp_module stubs, load engine_bridge fresh
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eb():
    """Load ADDON/core/engine_bridge.py with stubbed bpy + cpp_module."""
    # If a prior import succeeded, drop it so the stubs take effect.
    for mod_name in list(sys.modules):
        if mod_name == 'bpy' or mod_name.startswith('texturesynth_testaddon'):
            del sys.modules[mod_name]

    # Stub bpy (only the attributes engine_bridge touches at import + run time).
    bpy_stub = types.ModuleType('bpy')
    ctx_stub = types.SimpleNamespace()
    ctx_stub.scene = types.SimpleNamespace()       # _get_resolution falls back to 1024
    ctx_stub.space_data = None
    ctx_stub.window_manager = types.SimpleNamespace(windows=[])
    bpy_stub.context = ctx_stub
    data_stub = types.SimpleNamespace(node_groups=[], images=types.SimpleNamespace(get=lambda n: None, new=lambda *a, **k: None))
    bpy_stub.data = data_stub
    bpy_stub.types = types.SimpleNamespace(Node=object)
    sys.modules['bpy'] = bpy_stub

    # Build a synthetic package so `from . import cpp_module` resolves.
    pkg = types.ModuleType('texturesynth_testaddon')
    pkg.__path__ = []
    sys.modules['texturesynth_testaddon'] = pkg

    core_pkg = types.ModuleType('texturesynth_testaddon.core')
    core_pkg.__path__ = []
    sys.modules['texturesynth_testaddon.core'] = core_pkg

    # cpp_module stub.
    cpp_stub = types.ModuleType('texturesynth_testaddon.core.cpp_module')
    cpp_stub.get_core = lambda: FakeCore()
    cpp_stub.get_engine = lambda: None
    cpp_stub.is_loaded = lambda: True
    sys.modules['texturesynth_testaddon.core.cpp_module'] = cpp_stub
    core_pkg.cpp_module = cpp_stub

    # logging stub (engine_bridge does `from . import logging as _tslog`).
    log_stub = types.ModuleType('texturesynth_testaddon.core.logging')
    class _FakeLogger:
        def error(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
    log_stub._tslog = _FakeLogger()
    def _is_enabled_for(level): return False
    log_stub.is_enabled_for = _is_enabled_for
    sys.modules['texturesynth_testaddon.core.logging'] = log_stub
    core_pkg.logging = log_stub

    # Load engine_bridge.py from disk under the synthetic package name.
    import pathlib
    src_path = pathlib.Path(__file__).resolve().parents[2] / 'ADDON' / 'core' / 'engine_bridge.py'
    spec = importlib.util.spec_from_file_location(
        'texturesynth_testaddon.core.engine_bridge', src_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['texturesynth_testaddon.core.engine_bridge'] = mod
    core_pkg.engine_bridge = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Test 1: fingerprint must change when A/B edges are swapped
# --------------------------------------------------------------------------

def _build_two_producer_blend_tree():
    """simplex -> blend.a, color_const -> blend.b. Returns (tree, ids)."""
    tree = FakeTree()
    simplex  = _make_node(tree, 'Simplex',     101, 'simplex',     0, 1)
    color    = _make_node(tree, 'Color',       102, 'color_const', 0, 1)
    blend    = _make_node(tree, 'Blend',       103, 'blend',
                          n_inputs=3, input_names=['mask', 'a', 'b'],
                          n_outputs=1)
    out      = _make_node(tree, 'Output',      104, sv_type=None, bl_idname='TS_Output_Node',
                          n_inputs=1, input_names=['color'], n_outputs=0)
    # simplex -> blend.a (input idx 1), color -> blend.b (input idx 2)
    tree.connect(simplex, 0, blend, 1)
    tree.connect(color,   0, blend, 2)
    # blend -> output
    tree.connect(blend,   0, out,   0)
    out.select = True       # Output is the active/selected node
    ids = {'simplex': 101, 'color': 102, 'blend': 103, 'out': 104}
    return tree, ids


def test_fingerprint_changes_on_ab_edge_swap(eb):
    """Swapping which producer feeds blend.a vs blend.b MUST change the fingerprint."""
    tree, ids = _build_two_producer_blend_tree()
    fp_orig = eb._active_subgraph_fingerprint(tree)

    # Swap: disconnect both producers, rewire crossed.
    simplex = tree.nodes.get('Simplex')
    color   = tree.nodes.get('Color')
    blend   = tree.nodes.get('Blend')
    eb_tree = tree  # alias
    eb_tree.disconnect_all_to(blend, 1)
    eb_tree.disconnect_all_to(blend, 2)
    # Now simplex -> blend.b (idx 2), color -> blend.a (idx 1)
    eb_tree.connect(simplex, 0, blend, 2)
    eb_tree.connect(color,   0, blend, 1)

    fp_swap = eb._active_subgraph_fingerprint(tree)
    print(f"\n  fp_orig link part: {fp_orig[1]}")
    print(f"  fp_swap link part: {fp_swap[1]}")
    assert fp_orig != fp_swap, (
        "Swapping blend A/B edges did NOT change the topology fingerprint. "
        "evaluation.py would never request_topology_update() on a swap -> "
        "TS_Preview2D stays stale. This is the swap symptom's root cause."
    )


# --------------------------------------------------------------------------
# Test 2: active node drives root when Output is disconnected
# --------------------------------------------------------------------------

def test_resolve_root_picks_active_node_when_output_disconnected(eb):
    """Output node present but no input link -> root must be the active blend node."""
    tree, ids = _build_two_producer_blend_tree()
    # Disconnect blend->output so output_has_inputs becomes False.
    out = tree.nodes.get('Output')
    blend = tree.nodes.get('Blend')
    tree.disconnect_all_to(out, 0)

    # Make blend the active node.
    for n in tree.nodes:
        n.select = (n is blend)

    root, name_to_id, id_to_name, order, reachable = eb._resolve_preview_root(tree)
    assert root is not None, "root is None with an active TS node present"
    assert root.name == 'Blend', (
        f"Expected root=Blend (the active node), got root={root.name!r}. "
        "When Output is disconnected the active node should drive the preview."
    )
    assert ids['blend'] in reachable, "blend not in reachable subgraph"


# --------------------------------------------------------------------------
# Test 3: output_src_id follows the active blend node (two-blend scenario)
# --------------------------------------------------------------------------

def _build_two_blend_tree():
    """Two independent blend branches both feeding Output. Clicking blend#2
    must route output_src_id to blend#2, not blend#1."""
    tree = FakeTree()
    a1   = _make_node(tree, 'Src1', 201, 'simplex',     0, 1)
    a2   = _make_node(tree, 'Src2', 202, 'color_const', 0, 1)
    b1   = _make_node(tree, 'Src3', 203, 'simplex',     0, 1)
    b2   = _make_node(tree, 'Src4', 204, 'color_const', 0, 1)
    blend1 = _make_node(tree, 'Blend1', 205, 'blend',
                        n_inputs=3, input_names=['mask', 'a', 'b'], n_outputs=1)
    blend2 = _make_node(tree, 'Blend2', 206, 'blend',
                        n_inputs=3, input_names=['mask', 'a', 'b'], n_outputs=1)
    out  = _make_node(tree, 'Output', 207, sv_type=None, bl_idname='TS_Output_Node',
                      n_inputs=2, input_names=['color', 'alt'], n_outputs=0)
    # Branch 1: a1->blend1.a, a2->blend1.b, blend1->output[0]
    tree.connect(a1, 0, blend1, 1)
    tree.connect(a2, 0, blend1, 2)
    tree.connect(blend1, 0, out, 0)
    # Branch 2: b1->blend2.a, b2->blend2.b, blend2->output[1]
    tree.connect(b1, 0, blend2, 1)
    tree.connect(b2, 0, blend2, 2)
    tree.connect(blend2, 0, out, 1)
    return tree, {'a1':201,'a2':202,'b1':203,'b2':204,'blend1':205,'blend2':206,'out':207}


def test_output_src_id_follows_active_blend_two(eb):
    """Selecting blend#2 must make output_src_id resolve to blend#2's id."""
    tree, ids = _build_two_blend_tree()
    blend2 = tree.nodes.get('Blend2')
    for n in tree.nodes:
        n.select = (n is blend2)

    # Patch out the engine side-effects; we only care about the graph the
    # bridge WOULD submit. _build_graph_and_params calls cpp_module.get_core()
    # (already stubbed to FakeCore) so RecordingGraph captures everything.
    graph, pc, node_params = eb._build_graph_and_params(tree)
    assert graph is not None, "_build_graph_and_params returned None"
    print(f"\n  output_node id: {graph.output_node}")
    print(f"  output_targets: {graph.output_targets}")
    print(f"  connections: {graph.connections}")
    assert graph.output_node == ids['blend2'], (
        f"Expected output_node=blend2 (id {ids['blend2']}), got {graph.output_node}. "
        "Active blend#2 not taking over the output -> 'only one blend updates' symptom."
    )


# --------------------------------------------------------------------------
# Test 4: connections reflect the actual A/B socket assignment
# --------------------------------------------------------------------------

def test_connections_record_ab_socket_order(eb):
    """The RecordingGraph must capture simplex->blend.a(1), color->blend.b(2)."""
    tree, ids = _build_two_producer_blend_tree()
    graph, _, _ = eb._build_graph_and_params(tree)
    assert graph is not None
    # Expect (simplex=101 -> blend=103, di=1) and (color=102 -> blend=103, di=2)
    blend_connections = [c for c in graph.connections if c[2] == ids['blend']]
    print(f"\n  blend connections: {blend_connections}")
    socket_map = {c[0]: c[3] for c in blend_connections}   # src_id -> dst socket idx
    assert socket_map.get(ids['simplex']) == 1, (
        f"simplex should land on blend socket 1 (a), got {socket_map.get(ids['simplex'])}"
    )
    assert socket_map.get(ids['color']) == 2, (
        f"color should land on blend socket 2 (b), got {socket_map.get(ids['color'])}"
    )
