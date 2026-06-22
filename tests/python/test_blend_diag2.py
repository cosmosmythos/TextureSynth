"""Diagnose why Python gets byte-identical output for different mask values."""
import time
import numpy as np
import texturesynth_core as tc


def _build_blend_graph():
    g = tc.Graph()
    g.add_node(1, "simplex")
    g.add_node(2, "value")
    g.add_node(3, "blend")
    g.add_connection(1, 0, 3, 1)
    g.add_connection(2, 0, 3, 2)
    g.set_output(3)
    return g


def _render_blend_debug(engine, mask_value, label):
    g = _build_blend_graph()
    gen = engine.set_graph(g)
    assert gen != 0, f"set_graph failed: {engine.last_error_record().message!r}"
    print(f"  [{label}] set_graph gen={gen} installed={engine.installed_generation()}")

    engine.update_node_params_by_id(1, [8.0, 5.0, 2.0, 0.5, 0.0, 0.0, 43.0])
    engine.update_node_params_by_id(2, [8.0, 5.0, 2.0, 0.5, 0.0, 99.0])
    engine.update_node_params_by_id(3, [0.0, mask_value])
    print(f"  [{label}] params pushed: mask={mask_value}")

    for i in range(40):
        engine.poll_pending_compiles()
        if engine.has_pipeline():
            break
        time.sleep(0.05)
    print(f"  [{label}] pipeline ready after {i+1} polls")

    pc = tc.PushConstants()
    pc.resolution_x = 128
    pc.resolution_y = 128
    pc.seed = 42
    pc.time = 0.0

    ticket = engine.submit_render(pc, engine.installed_generation())
    print(f"  [{label}] submit ticket={ticket}")

    for j in range(60):
        r = engine.poll_readback()
        if r is not None:
            arr, ret_gen = r
            print(f"  [{label}] readback gen={ret_gen} shape={arr.shape} mean={arr.mean():.6f}")
            return arr
        time.sleep(0.05)
    assert False, "readback timed out"


def test_diagnostic(engine):
    px0 = _render_blend_debug(engine, 0.0, "A")
    px1 = _render_blend_debug(engine, 1.0, "B")

    diff = float(np.sum(np.abs(px0 - px1)))
    print(f"\n  DIFF: {diff:.4f}")
    print(f"  px0[64,64] = {px0[64,64]}")
    print(f"  px1[64,64] = {px1[64,64]}")
    print(f"  bytes_equal = {px0.tobytes() == px1.tobytes()}")

    assert diff > 1.0, f"Identical output (diff={diff})"
