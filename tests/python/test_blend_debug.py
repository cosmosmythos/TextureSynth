"""Blend mask debug — dumps diagnostic info to find why diff=0."""
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


def _render_blend(engine, mask_value):
    g = _build_blend_graph()
    gen = engine.set_graph(g)
    assert gen != 0, f"set_graph failed: {engine.last_error_record().message!r}"

    engine.update_node_params_by_id(1, [8.0, 5.0, 2.0, 0.5, 0.0, 0.0, 43.0])
    engine.update_node_params_by_id(2, [8.0, 5.0, 2.0, 0.5, 0.0, 99.0])
    engine.update_node_params_by_id(3, [0.0, mask_value])

    # Check param layout
    layout = engine.param_layout()
    total = engine.total_param_floats()

    for _ in range(40):
        engine.poll_pending_compiles()
        if engine.has_pipeline():
            break
        time.sleep(0.05)
    assert engine.has_pipeline(), "pipeline not ready"

    pc = tc.PushConstants()
    pc.resolution_x = 128
    pc.resolution_y = 128
    pc.seed = 42
    pc.time = 0.0

    ticket = engine.submit_render(pc, engine.installed_generation())
    assert ticket > 0

    for _ in range(60):
        r = engine.poll_readback()
        if r is not None:
            arr, ret_gen = r
            return arr, layout, total, ret_gen
        time.sleep(0.05)
    pytest.fail("poll_readback timed out")


import pytest

def test_debug_mask(engine):
    px0, layout0, total0, gen0 = _render_blend(engine, 0.0)
    px1, layout1, total1, gen1 = _render_blend(engine, 1.0)

    print(f"\n  layout0={layout0} total0={total0} gen0={gen0}")
    print(f"  layout1={layout1} total1={total1} gen1={gen1}")
    print(f"  px0 shape={px0.shape} dtype={px0.dtype} mean={px0.mean():.6f} min={px0.min():.6f} max={px0.max():.6f}")
    print(f"  px1 shape={px1.shape} dtype={px1.dtype} mean={px1.mean():.6f} min={px1.min():.6f} max={px1.max():.6f}")

    diff_map = np.abs(px0 - px1)
    print(f"  diff_map mean={diff_map.mean():.6f} max={diff_map.max():.6f} nonzero={np.count_nonzero(diff_map)}")

    # Check if arrays are the exact same object
    print(f"  same_array={px0 is px1}")
    print(f"  bytes_equal={px0.tobytes() == px1.tobytes()}")

    # Sample a few pixels
    h, w = px0.shape[:2]
    for y in [0, h//4, h//2, 3*h//4, h-1]:
        for x in [0, w//4, w//2, 3*w//4, w-1]:
            v0 = px0[y, x]
            v1 = px1[y, x]
            if not np.array_equal(v0, v1):
                print(f"  [{y},{x}] mask0={v0} mask1={v1} DIFF")
    print(f"  DONE")
