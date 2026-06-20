"""End-to-end bake test: Output targets -> texturesynth.bake -> image data-blocks.

Builds:  Perlin -> Output (two targets: "Albedo", "Roughness")
Wires perlin into both targets, runs the bake operator, and verifies:
  - the operator returned FINISHED
  - image data-blocks named "Albedo" and "Roughness" exist
  - they have non-zero pixel content (the bake actually rendered)

Template for multi-target / unlinked-target / resolution bake tests.

CONTEXT GOTCHA: texturesynth.output_target_add reads context.active_node,
which in MCP headless runs reflects the user's real selection, NOT what we
set via tree.nodes.active. So this script adds Output sockets via the DIRECT
API (node.inputs.new) instead of the operator -- robust under MCP.

WARNING: clears the tree + removes stale Albedo/Roughness images. Run only
on a throwaway graph; for the user's existing setup use introspect.py first.

texturesynth.bake itself does NOT need context.active_node -- it resolves
the Output node via engine_bridge._active_output_node (any TS tree).

Send via Blender MCP execute_code. Output comes back as captured stdout.
"""
import sys
import importlib
import numpy as np


def _ts_pkg():
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        if name in sys.modules:
            return importlib.import_module(name)
    for mod in list(sys.modules.values()):
        core = getattr(mod, "core", None)
        if core is not None and hasattr(core, "cpp_module"):
            return mod
    raise RuntimeError("TextureSynth addon not loaded")


def _img_stats(name):
    import bpy
    img = bpy.data.images.get(name)
    if img is None:
        return None
    w, h = img.size
    if w == 0 or h == 0:
        return (w, h, None)
    pix = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(pix)
    return (w, h, float(pix.mean()))


def main():
    import bpy
    ts = _ts_pkg()
    core = ts.core
    engine_bridge = core.engine_bridge
    engine = core.cpp_module.get_engine()
    if engine is None:
        print("FAIL: engine not initialized")
        return False

    tree = engine_bridge._find_node_tree()
    if tree is None:
        print("FAIL: no TS tree -- run build_graph.py first")
        return False

    for n in list(tree.nodes):
        tree.nodes.remove(n)
    for l in list(tree.links):
        tree.links.remove(l)
    for stale in ("Albedo", "Roughness"):
        img = bpy.data.images.get(stale)
        if img is not None:
            bpy.data.images.remove(img)

    perlin = tree.nodes.new('TS_Perlin_Node')
    perlin.location = (-300, 0)
    perlin.period = 8.0

    out = tree.nodes.new('TS_Output_Node')
    out.location = (100, 0)

    # Output node starts with one target ("Base Color"). Rename it, then add a
    # SECOND target via the direct socket API (NOT bpy.ops, which needs
    # context.active_node that MCP won't honor for our node).
    out.inputs[0].name = "Albedo"
    out.inputs.new('TS_BakeTargetSocketType', "Roughness")

    tree.links.new(perlin.outputs[0], out.inputs[0])   # Albedo
    tree.links.new(perlin.outputs[0], out.inputs[1])   # Roughness

    # texturesynth.bake resolves the Output node itself; no context override
    # needed. It submits the graph, waits for compile, and bakes synchronously.
    res = bpy.ops.texturesynth.bake()
    print(f"bake operator returned: {res}")
    if 'FINISHED' not in res:
        print(f"FAIL: bake did not finish. engine last_error: "
              f"{engine.last_error()!r}")
        return False

    all_ok = True
    for name in ("Albedo", "Roughness"):
        stats = _img_stats(name)
        if stats is None:
            print(f"FAIL: image {name!r} not created")
            all_ok = False
            continue
        w, h, mean = stats
        print(f"target {name:10s}: {w}x{h} mean={mean}")
        if mean is None:
            print(f"FAIL: {name!r} has zero-size pixels")
            all_ok = False
            continue
        # A baked perlin should be well above 0. Near-black => target wasn't
        # linked or the bake silently failed.
        if mean < 1e-4:
            print(f"FAIL: {name!r} baked to near-black")
            all_ok = False

    if all_ok:
        print("PASS: both targets baked with non-zero content")
    return all_ok


ok = main()
