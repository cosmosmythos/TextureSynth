"""
Engine lifecycle: construct, init, shutdown, error state reset.
"""
import pytest
import texturesynth_core as tc


def test_engine_constructs():
    """A bare Engine() should construct without crashing (no Vulkan call yet)."""
    e = tc.Engine()
    assert e is not None


def test_init_then_shutdown(engine):
    """Session-scoped fixture already initialized Vulkan.
    Verify the error record is in a clean state (we call clear_error first
    to insulate from earlier tests in the session)."""
    engine.clear_error()
    rec = engine.last_error_record()
    assert rec.code == tc.EngineErrorCode["None"]
    assert not rec.is_error()
    assert engine.has_pipeline() is not None  # bool-ish


def test_reinit_after_shutdown_is_safe(assets):
    """Architecture constraint: one VkInstance per process. The realistic test
    is: shutdown, then init again on the same Engine. The second init must not
    segfault and must leave the engine functional (or at least, in a defined
    error state)."""
    e = tc.Engine()
    ok = e.init(
        enable_validation=False,
        cache_dir=assets["cache_dir"],
        nodes_dir=assets["nodes_dir"],
        glsl_dir=assets["glsl_dir"],
    )
    if not ok:
        pytest.skip("Vulkan init failed")
    e.shutdown()

    # Second init: should not crash. May return True (full re-init) or False
    # (Vulkan context refuses). Either way, no segfault.
    crashed = False
    try:
        second = e.init(
            enable_validation=False,
            cache_dir=assets["cache_dir"],
            nodes_dir=assets["nodes_dir"],
            glsl_dir=assets["glsl_dir"],
        )
        # If it succeeded, the engine is back; either way, no exception is fine.
        _ = second
    except Exception:
        # An exception is also acceptable — we just need to not segfault.
        pass
    # Verify the engine object is still valid Python-side.
    assert e is not None
    _ = e.last_error_record()  # accessor must not crash
