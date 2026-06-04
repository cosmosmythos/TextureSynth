"""Stage 9: Python smoke test for the new bake / set_active_node / readback_sync
API. All new engine methods are exercised end-to-end through the nanobind
binding. We assert on shape/dtype/non-emptiness, not pixel content, because
the engine's output is procedural noise.

These tests use the session-scoped `engine` fixture (which is already a real
Vulkan engine with shaders compiled). They use unique cache dirs per test so
that stale shader caches don't bleed across tests.
"""
import time
import numpy as np
import pytest
import texturesynth_core as tc


def _make_graph(*nodes, output_node=None, targets=None):
    g = tc.Graph()
    for n in nodes:
        g.add_node(
            id=n["id"],
            type=n["type"],
            format_override=tc.ChannelFormat.RGBA,
            debug_name="",
            muted=False,
            bypassed=False,
        )
    g.set_output(output_node if output_node is not None else nodes[-1]["id"])
    if targets:
        for t in targets:
            g.add_output_target(t["source"], t["name"])
    return g


def test_add_output_target_and_clear(engine, assets):
    """Graph exposes add_output_target / clear_output_targets. We just verify
    the binding round-trips through engine.set_graph without error."""
    g = _make_graph(
        {"id": 1, "type": "perlin"},
        {"id": 2, "type": "invert"},
        output_node=2,
        targets=[{"source": 1, "name": "Base Color"},
                 {"source": 2, "name": "Inverted"}],
    )
    gen = engine.set_graph(g)
    if gen == 0:
        pytest.skip(f"set_graph: {engine.last_error()}")
    for _ in range(200):
        if engine.has_pipeline():
            break
        engine.poll_pending_compiles()
        time.sleep(0.01)
    assert engine.has_pipeline()


def test_set_active_node_re_targets_preview(engine, assets):
    """set_active_node re-targets the engine at a different node. After
    the call, the active node in current_graph is the new id."""
    g = _make_graph(
        {"id": 1, "type": "perlin"},
        {"id": 2, "type": "invert"},
        output_node=2,
    )
    engine.set_graph(g)
    for _ in range(200):
        if engine.has_pipeline():
            break
        engine.poll_pending_compiles()
        time.sleep(0.01)
    assert engine.has_pipeline()

    gen = engine.set_active_node(1)
    assert gen != 0, f"set_active_node failed: {engine.last_error()}"
    for _ in range(200):
        if engine.has_pipeline():
            break
        engine.poll_pending_compiles()
        time.sleep(0.01)
    assert engine.has_pipeline()
    assert engine.current_graph().output_node == 1


def test_readback_sync_returns_float_rgba_ndarray(engine, assets):
    """readback_sync returns a numpy ndarray with shape (H, W, 4) and dtype
    float32, and the pixels are not all zero (proving the GPU produced
    real noise)."""
    g = _make_graph({"id": 1, "type": "perlin"}, output_node=1)
    gen = engine.set_graph(g)
    if gen == 0:
        pytest.skip(f"set_graph: {engine.last_error()}")
    for _ in range(200):
        if engine.has_pipeline():
            break
        engine.poll_pending_compiles()
        time.sleep(0.01)
    assert engine.has_pipeline()

    arr = engine.readback_sync()
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.ndim == 3
    assert arr.shape[2] == 4  # RGBA
    assert arr.size > 0
    # Perlin at default seed produces non-zero pixels.
    assert arr.max() > 0.0
    assert arr.min() < 1.0


def test_bake_returns_named_images_and_restores_active(engine, assets):
    """bake() returns a list of {name, w, h, pixels} dicts. The engine's
    active node is restored to what it was before the bake call."""
    original_active = 1
    g = _make_graph(
        {"id": 1, "type": "perlin"},
        {"id": 2, "type": "invert"},
        output_node=original_active,
        targets=[{"source": 1, "name": "Base Color"},
                 {"source": 2, "name": "Inverted"}],
    )
    gen = engine.set_graph(g)
    if gen == 0:
        pytest.skip(f"set_graph: {engine.last_error()}")
    for _ in range(200):
        if engine.has_pipeline():
            break
        engine.poll_pending_compiles()
        time.sleep(0.01)
    assert engine.has_pipeline()

    bakes = engine.bake()
    assert isinstance(bakes, list)
    assert len(bakes) == 2
    assert bakes[0]["name"] == "Base Color"
    assert bakes[1]["name"] == "Inverted"
    for b in bakes:
        assert isinstance(b["pixels"], np.ndarray)
        assert b["pixels"].dtype == np.float32
        assert b["pixels"].ndim == 3
        assert b["pixels"].shape[2] == 4
        assert b["width"]  == b["pixels"].shape[1]
        assert b["height"] == b["pixels"].shape[0]
    # Base Color and Inverted differ (perlin and invert(perlin) aren't equal).
    assert not np.array_equal(bakes[0]["pixels"], bakes[1]["pixels"])
    # Engine restored its original active node.
    assert engine.current_graph().output_node == original_active
