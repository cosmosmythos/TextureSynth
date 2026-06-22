"""Blend mask correctness — compares blend output against itself with different mask values."""
import time
import numpy as np
import pytest
import texturesynth_core as tc


def _build_blend_graph():
    g = tc.Graph()
    g.add_node(1, "simplex")
    g.add_node(2, "value")
    g.add_node(3, "blend")
    g.add_connection(1, 0, 3, 1)  # simplex -> blend.a
    g.add_connection(2, 0, 3, 2)  # value   -> blend.b
    g.set_output(3)
    return g


def _render_blend(engine, mask_value):
    g = _build_blend_graph()
    gen = engine.set_graph(g)
    assert gen != 0, f"set_graph failed: {engine.last_error_record().message!r}"

    engine.update_node_params_by_id(1, [8.0, 5.0, 2.0, 0.5, 0.0, 0.0, 43.0])
    engine.update_node_params_by_id(2, [8.0, 5.0, 2.0, 0.5, 0.0, 99.0])
    engine.update_node_params_by_id(3, [0.0, mask_value])

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
            arr, _ = r
            return arr
        time.sleep(0.05)
    pytest.fail("poll_readback timed out")


def test_mask_zero_vs_mask_one(engine):
    """mask=0 and mask=1 MUST produce different output."""
    px0 = _render_blend(engine, 0.0)
    px1 = _render_blend(engine, 1.0)
    diff = float(np.sum(np.abs(px0 - px1)))
    print(f"\n  mask=0 vs mask=1: diff={diff:.4f}")
    assert diff > 1.0, f"mask=0 and mask=1 identical (diff={diff})"


def test_mask_zero_vs_mask_half(engine):
    """mask=0 and mask=0.5 MUST produce different output."""
    px0 = _render_blend(engine, 0.0)
    px_half = _render_blend(engine, 0.5)
    diff = float(np.sum(np.abs(px0 - px_half)))
    print(f"\n  mask=0 vs mask=0.5: diff={diff:.4f}")
    assert diff > 1.0, f"mask=0 and mask=0.5 identical (diff={diff})"


def test_mask_one_vs_mask_half(engine):
    """mask=1 and mask=0.5 MUST produce different output."""
    px1 = _render_blend(engine, 1.0)
    px_half = _render_blend(engine, 0.5)
    diff = float(np.sum(np.abs(px1 - px_half)))
    print(f"\n  mask=1 vs mask=0.5: diff={diff:.4f}")
    assert diff > 1.0, f"mask=1 and mask=0.5 identical (diff={diff})"
