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


# Stage 1 / 0.2: Python can ask "how much VRAM did the engine actually use?"
def test_vma_stats_exposes_all_fields(engine):
    """get_vma_stats() returns a dict with all fields documented in
    VmaStatsReport (see src/engine/ResourceManager.hpp). After at least
    one set_graph the live count must be > 0 and VMA must report a
    non-zero allocation."""
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

    stats = engine.get_vma_stats()
    # Required keys (per VmaStatsReport v2 in ResourceManager.hpp).
    expected_keys = {
        # Logical view
        "node_resource_count", "node_resource_bytes",
        "retired_count", "retired_bytes",
        # VMA totals
        "vma_block_bytes", "vma_allocation_bytes",
        "vma_unused_range_bytes",
        # v2: configured warning threshold (real GPU numbers are gpu_*)
        "warning_threshold_bytes",
        # Back-compat alias for the old `budget_bytes` field.
        "budget_bytes",
        # v2: real GPU numbers (sum of device-local heaps)
        "gpu_budget_bytes", "gpu_usage_bytes", "gpu_pressure",
        # v2: per-heap breakdown list
        "heap_stats",
        # Aggregation helper
        "aliasing_efficiency",
    }
    assert expected_keys.issubset(stats.keys()), (
        f"missing keys: {expected_keys - set(stats.keys())}"
    )
    assert stats["node_resource_count"] >= 1
    assert stats["node_resource_bytes"] > 0
    assert stats["vma_block_bytes"] > 0
    assert stats["vma_allocation_bytes"] > 0
    # aliasing_efficiency = vma_allocation_bytes / node_resource_bytes.
    # In a clean ResourceManager (no other VMA allocations) the ratio is
    # exactly 1.0. In the live engine the param SSBO ring, output storage
    # image, and bindless bookkeeping add to VMA's totals, so the ratio
    # is > 1.0 (VMA has more bytes than our resources logically need).
    # A value < 1.0 here would mean aliasing is happening (Stage 6
    # transient aliasing). The smoke test just checks the math is sane.
    eff = stats["aliasing_efficiency"]
    assert eff > 0.0, f"aliasing_efficiency must be positive, got {eff}"
    assert eff < 1e6, f"aliasing_efficiency unreasonably large: {eff}"

    # warning_threshold_bytes == budget_bytes (back-compat alias).
    assert stats["warning_threshold_bytes"] == stats["budget_bytes"], (
        "warning_threshold_bytes and budget_bytes should be equal -- "
        "budget_bytes is a back-compat alias for warning_threshold_bytes"
    )
    # warning_threshold_bytes reflects the configured 1 GB default.
    assert stats["warning_threshold_bytes"] == 1024 * 1024 * 1024


# Stage 1 / 0.2 v2: the stats report now exposes the real GPU VRAM
# budget (sum of device-local heaps) and a per-heap breakdown. On any
# conforming Vulkan implementation there is at least one device-local
# heap (spec requirement), and VMA must report a non-zero budget for
# it when the allocator was created with VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT
# (which VulkanContext.cpp does).
def test_vma_stats_exposes_real_gpu_budget_and_heap_breakdown(engine):
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

    stats = engine.get_vma_stats()

    # Real GPU numbers: at least one device-local heap exists, so
    # gpu_budget_bytes > 0. If the driver failed to populate the budget
    # extension AND VMA's 80% fallback is exactly 0, this would be 0 --
    # but in practice that's never the case.
    assert stats["gpu_budget_bytes"] > 0, (
        "gpu_budget_bytes must be > 0 -- the device-local heap budget "
        "is the real GPU VRAM. Zero means either no device-local heap "
        "(invalid Vulkan) or VMA budget extension failed."
    )
    # gpu_pressure is clamped 0..1.
    assert 0.0 <= stats["gpu_pressure"] <= 1.0

    # Per-heap breakdown: at least one entry (any Vulkan implementation
    # has at least one memory heap). On a discrete GPU there are usually
    # 2 (DEVICE_LOCAL + HOST_VISIBLE). On UMA there's 1.
    assert len(stats["heap_stats"]) >= 1
    saw_device_local = False
    total_allocations = 0
    for h in stats["heap_stats"]:
        assert "index" in h
        assert "is_device_local" in h
        assert "label" in h
        assert h["label"] in ("DEVICE_LOCAL", "HOST")
        assert "budget_bytes" in h
        assert "usage_bytes" in h
        assert "vma_allocation_count" in h
        assert "pressure" in h
        assert 0.0 <= h["pressure"] <= 1.0
        total_allocations += h["vma_allocation_count"]
        if h["is_device_local"]:
            saw_device_local = True
    assert saw_device_local, "no device-local heap in the breakdown"
    # The aggregate across all heaps must include our allocations.
    # (Don't assume *which* heap they went into -- VMA's AUTO strategy
    # may place TILING_OPTIMAL images in a device-local or host-visible
    # heap depending on driver heuristics, and we don't want to assert
    # a specific placement that varies by GPU vendor.)
    assert total_allocations >= 1, (
        "no VMA allocations found in any heap -- images not allocated?"
    )


# Stage 1 / 0.3: Python `update_node_params_by_name` between two renders
# must produce different output. Before the fix, the C++ side only wrote
# to NodeResource::is_dirty (Layer 3) but never seeded the DirtySet
# (Layer 2), so the next submit would early-exit on an empty dirty set
# and re-publish the prior frame. The C++ ParamUpdateTriggersRedispatch
# test is the ground truth; this Python version guards the binding.
@pytest.mark.timeout(15)
def test_param_update_triggers_redispatch(engine):
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
    pc.seed = 1
    pc.time = 0.0

    # Frame 1: defaults
    t1 = engine.submit_render(pc, engine.installed_generation())
    assert t1 > 0
    p1 = None
    for _ in range(60):
        r = engine.poll_readback()
        if r is not None:
            p1, _ = r
            break
        time.sleep(0.05)
    assert p1 is not None and p1.size > 0

    # Change the perlin seed param. perlin.node.json has 8 params ending
    # in "seed" (default 0). Setting it to 1234 produces visibly
    # different noise.
    engine.update_node_params_by_name(1, {
        "period": 8.0, "octaves": 5.0, "lacunarity": 2.0,
        "roughness": 0.5, "speed": 0.0, "seed": 1234.0,
    })

    # Frame 2: must be different from frame 1
    t2 = engine.submit_render(pc, engine.installed_generation())
    assert t2 > 0
    p2 = None
    for _ in range(60):
        r = engine.poll_readback()
        if r is not None:
            p2, _ = r
            break
        time.sleep(0.05)
    assert p2 is not None and p2.size > 0
    assert p1.shape == p2.shape

    # At least one pixel must differ.
    assert np.any(p1 != p2), (
        "param update did not trigger re-dispatch (p1 == p2 -- "
        "engine likely early-exited on empty dirty_set_)"
    )
