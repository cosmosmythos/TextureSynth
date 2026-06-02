"""
Render path: submit + poll_readback, including the no-pipeline error.
"""
import time
import numpy as np
import pytest
import texturesynth_core as tc


def test_submit_render_without_pipeline_raises(assets):
    """A fresh engine that has init'd but never had a graph must raise
    NoPipeline on submit. We use a fresh engine to avoid contamination from
    the session-scoped engine, which may have a live pipeline."""
    e = tc.Engine()
    ok = e.init(
        enable_validation=False,
        cache_dir=assets["cache_dir"],
        nodes_dir=assets["nodes_dir"],
        glsl_dir=assets["glsl_dir"],
    )
    if not ok:
        pytest.skip("Vulkan init failed")
    assert not e.has_pipeline()

    pc = tc.PushConstants()
    pc.resolution_x = 64
    pc.resolution_y = 64
    with pytest.raises(RuntimeError):
        e.submit_render(pc, 0)

    rec = e.last_error_record()
    assert rec.code == tc.EngineErrorCode.NoPipeline
    assert rec.phase == tc.EnginePhase.Submit
    e.shutdown()


def test_poll_readback_returns_none_when_empty(engine):
    """A fresh engine with no submissions should yield None."""
    res = engine.poll_readback()
    assert res is None


def test_stale_generation_raises(engine):
    """Submitting with a future/garbage generation should raise with
    StaleGeneration structured error."""
    g = tc.Graph()
    g.add_node(1, "perlin")
    g.set_output(1)
    gen = engine.set_graph(g)
    assert gen != 0
    for _ in range(40):
        engine.poll_pending_compiles()
        if engine.has_pipeline():
            break
        time.sleep(0.05)
    assert engine.has_pipeline()

    pc = tc.PushConstants()
    pc.resolution_x = 64
    pc.resolution_y = 64
    with pytest.raises(RuntimeError):
        engine.submit_render(pc, 99999)

    rec = engine.last_error_record()
    assert rec.code == tc.EngineErrorCode.StaleGeneration
    assert rec.graph_generation == 99999
    assert rec.phase == tc.EnginePhase.Submit


@pytest.mark.timeout(15)
def test_full_dispatch_returns_pixels(engine):
    """End-to-end: compile, submit, poll. Output must be a float32 RGBA array."""
    g = tc.Graph()
    g.add_node(1, "perlin")
    g.set_output(1)
    gen = engine.set_graph(g)
    assert gen != 0
    for _ in range(40):
        engine.poll_pending_compiles()
        if engine.has_pipeline():
            break
        time.sleep(0.05)

    pc = tc.PushConstants()
    pc.resolution_x = 64
    pc.resolution_y = 64
    pc.seed = 1234  # uint32_t in C++ — must be int

    ticket = engine.submit_render(pc, engine.installed_generation())
    assert ticket > 0, f"submit returned {ticket}"

    res = None
    for _ in range(60):
        res = engine.poll_readback()
        if res is not None:
            break
        time.sleep(0.05)
    assert res is not None, "poll_readback timed out"
    arr, gen_out = res
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.ndim == 3 and arr.shape[2] == 4
    # Perlin output should not be all zeros
    assert arr.max() > 0.0 or arr.min() < 0.0, "perlin output is suspiciously flat"
