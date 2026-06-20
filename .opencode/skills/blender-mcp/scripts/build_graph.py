"""Build a small TextureSynth graph programmatically and verify it compiles.

Constructs:  Perlin -> Invert -> Output
Forces a topology update, waits for the engine to accept the graph, and
reports generation + any compile error.

Template for any "does node X wire up correctly" test:
  - get the active tree (create if missing)
  - clear previous nodes   <-- DESTRUCTIVE: only run on a throwaway tree
  - create nodes by exact bl_idname
  - set props
  - wire links by socket name where ambiguous
  - request_topology_update() + submit_graph() + wait
  - read engine.last_error_record()

WARNING: this clears the tree. In Mode A (user has a graph open), do NOT
run this -- use introspect.py instead. Only run when the user wants a fresh
controlled test graph.

Send via Blender MCP execute_code. Output comes back as captured stdout.
"""
import sys
import importlib
import time


def _ts_pkg():
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        if name in sys.modules:
            return importlib.import_module(name)
    for mod in list(sys.modules.values()):
        core = getattr(mod, "core", None)
        if core is not None and hasattr(core, "cpp_module"):
            return mod
    raise RuntimeError("TextureSynth addon not loaded")


def _wait_for_ready(engine, gen, timeout=5.0, interval=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        engine.poll_pending_compiles()
        if gen and engine.is_generation_ready(gen):
            return True
        time.sleep(interval)
    return False


def main():
    import bpy
    ts = _ts_pkg()
    core = ts.core
    engine_bridge = core.engine_bridge
    evaluation = core.evaluation
    engine = core.cpp_module.get_engine()
    if engine is None:
        print("FAIL: engine not initialized -- run smoke_test.py first")
        return False

    tree = engine_bridge._find_node_tree()
    if tree is None:
        tree = bpy.data.node_groups.new("TS_Test", "TextureSynthTreeType")
        print(f"(created new tree {tree.name!r})")

    # Clean slate so prior state can't mask a failure.
    for n in list(tree.nodes):
        tree.nodes.remove(n)
    for l in list(tree.links):
        tree.links.remove(l)

    # --- Build: Perlin -> Invert -> Output ---
    perlin = tree.nodes.new('TS_Perlin_Node')
    perlin.location = (-400, 0)
    perlin.period = 8.0
    perlin.octaves = 5.0
    perlin.seed = 0.0

    invert = tree.nodes.new('TS_Invert_Node')
    invert.location = (-150, 0)

    out = tree.nodes.new('TS_Output_Node')
    out.location = (100, 0)

    # Invert socket 0 is `mask` (default 1.0); socket 1 is `color`.
    tree.links.new(perlin.outputs[0], invert.inputs['color'])
    tree.links.new(invert.outputs[0], out.inputs[0])

    # Drive the bridge directly for determinism (timer would also work).
    evaluation.request_topology_update()
    gen = engine_bridge.submit_graph()

    rec = engine.last_error_record()
    print(f"submitted generation : {gen}")
    print(f"engine error code    : {rec.code}  phase={rec.phase}")
    print(f"failed node id       : {engine.failed_node()}")
    print(f"error message        : {rec.message!r}")

    if gen == 0:
        print("FAIL: set_graph rejected the graph")
        return False

    if not _wait_for_ready(engine, gen):
        print(f"FAIL: pipeline not ready for gen {gen} within timeout")
        return False
    print("pipeline ready")

    for n in (perlin, invert):
        err = getattr(n, 'ts_compile_error', '')
        if err:
            print(f"FAIL: {n.bl_idname} compile error: {err!r}")
            return False

    print("PASS: Perlin -> Invert -> Output compiled")
    return True


ok = main()
