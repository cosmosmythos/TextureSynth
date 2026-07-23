"""Sweep a slider and confirm the engine re-dispatches + pixels change.

Builds a single Perlin node, submits, reads back pixels for seed=0,
then changes seed and reads back again. The two pixel sets MUST differ --
if they don't, the param update didn't reach the GPU (a common regression:
the bridge's param-hash dedupe, the eval timer sleeping, or a stale layout).

Template for "did my slider change actually take effect" tests.
Substitute node + param name + values for the specific behavior.

WARNING: clears the tree. Use only on a throwaway graph; for testing the
user's existing setup, run introspect.py + a non-destructive sweep that
restores the original value afterward (see the restore pattern below).

Key technique: drive engine_bridge directly (submit_graph +
update_params_only) instead of waiting on the timer, for determinism.

Send via Blender MCP execute_code. Output comes back as captured stdout.
"""
import sys
import importlib
import time
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


def _wait_landed(engine_bridge, engine, timeout=5.0, interval=0.03):
    """Force dispatch and block until a fresh frame is read back.

    Returns (pixels_hxw4, gen) or (None, 0) on timeout.
    """
    deadline = time.time() + timeout
    engine_bridge.update_params_only(force_submit=True)
    while time.time() < deadline:
        engine.poll_pending_compiles()
        result = engine.poll_readback()
        if result is not None:
            pixels, gen = result
            if gen and gen >= engine_bridge._last_applied_generation:
                return pixels, gen
        engine_bridge.update_params_only(force_submit=True)
        time.sleep(interval)
    return None, 0


def _stats(pixels):
    if pixels is None:
        return "none"
    return (f"mean={float(pixels.mean()):.4f} "
            f"std={float(pixels.std()):.4f} "
            f"nonzero={float((pixels != 0).mean()):.3f}")


def main():
    import bpy
    ts = _ts_pkg()
    core = ts.core
    engine_bridge = core.engine_bridge
    evaluation = core.evaluation
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

    perlin = tree.nodes.new('TS_Perlin_Node')
    perlin.location = (0, 0)
    perlin.period = 8.0
    perlin.seed = 0.0

    evaluation.request_topology_update()
    gen = engine_bridge.submit_graph()
    if gen == 0:
        print(f"FAIL: initial submit rejected: "
              f"{engine.last_error_record().message!r}")
        return False

    deadline = time.time() + 5.0
    while time.time() < deadline and not engine.is_generation_ready(gen):
        engine.poll_pending_compiles()
        time.sleep(0.03)
    if not engine.is_generation_ready(gen):
        print("FAIL: shader compile timed out")
        return False

    baseline, gen0 = _wait_landed(engine_bridge, engine)
    print(f"baseline (seed=0)    gen={gen0}: {_stats(baseline)}")
    if baseline is None:
        print("FAIL: no readback for baseline")
        return False

    # Sweep the seed. Save the original so this stays non-destructive if
    # someone reuses the pattern on the user's real node.
    original_seed = float(perlin.seed)
    perlin.seed = 12345.0
    evaluation.request_param_update()

    swept, gen1 = _wait_landed(engine_bridge, engine)
    print(f"swept   (seed=12345) gen={gen1}: {_stats(swept)}")
    if swept is None:
        print("FAIL: no readback after seed change")
        return False

    if baseline.shape != swept.shape:
        print(f"FAIL: shape mismatch {baseline.shape} vs {swept.shape}")
        return False
    diff = float(np.abs(
        baseline.astype(np.float64) - swept.astype(np.float64)).mean())
    print(f"mean abs pixel diff : {diff:.6f}")

    # Restore -- harmless here, but keep the habit for non-destructive sweeps.
    perlin.seed = original_seed

    if diff < 1e-4:
        print("FAIL: pixels identical after seed change -- param update "
              "did not dispatch")
        return False

    print("PASS: seed change produced different output")
    return True


ok = main()
