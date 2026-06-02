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
    # The new state-machine accessors must be exposed and correct on a
    # session-scoped engine that is in use.
    assert engine.is_ready() is True
    assert int(engine.engine_state()) == 2  # EngineState::Ready == 2


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
        # An exception is also acceptable -- we just need to not segfault.
        pass
    # Verify the engine object is still valid Python-side.
    assert e is not None
    _ = e.last_error_record()  # accessor must not crash


# ---------------------------------------------------------------------------
# Stage 4+5+6 -- Engine lifecycle + guards, exercised from Python.
#
# These tests use a private engine (not the session fixture) so the session
# engine stays Ready for the rest of the suite. Each test scopes its own
# tc.Engine() and shuts it down before returning. If the Vulkan loader on
# this machine refuses a second VkInstance in one process, the test skips.
# ---------------------------------------------------------------------------


def _init_or_skip(e, assets):
    ok = e.init(
        enable_validation=False,
        cache_dir=assets["cache_dir"],
        nodes_dir=assets["nodes_dir"],
        glsl_dir=assets["glsl_dir"],
    )
    if not ok:
        pytest.skip(f"Vulkan init failed: {e.last_error()!r}")


def test_init_twice_is_idempotent(assets):
    """init() called on a Ready engine must return True and not change state."""
    e = tc.Engine()
    _init_or_skip(e, assets)
    assert e.is_ready() is True
    # Second init: must succeed and stay Ready.
    assert e.init(
        enable_validation=False,
        cache_dir=assets["cache_dir"],
        nodes_dir=assets["nodes_dir"],
        glsl_dir=assets["glsl_dir"],
    ) is True
    assert e.is_ready() is True
    e.shutdown()
    assert e.is_ready() is False
    assert int(engine_state_unwrap(e)) == 4  # EngineState::ShutDown == 4


def test_shutdown_is_idempotent(assets):
    """shutdown() called twice on the same engine must not crash."""
    e = tc.Engine()
    _init_or_skip(e, assets)
    e.shutdown()
    e.shutdown()  # second call must be a no-op
    assert e.is_ready() is False


def test_submit_after_shutdown_raises(assets):
    """Every mutator on a shut-down engine must throw RuntimeError and
    populate last_error_record with UseAfterShutdown."""
    e = tc.Engine()
    _init_or_skip(e, assets)
    e.shutdown()
    assert e.is_ready() is False

    # Clear any pre-shutdown error so the assertion below is unambiguous.
    e.clear_error()

    # set_graph: must throw (matches submit_render's existing error pattern).
    with pytest.raises(RuntimeError):
        e.set_graph(tc.Graph())

    # All other mutators: their bindings lambda now calls check_engine_ready,
    # which throws RuntimeError. We don't have to enumerate every one --
    # the C++ gtest covers the C++ side exhaustively. Here we just confirm
    # the user-facing behavior is "throws, error record populated".
    rec = e.last_error_record()
    assert rec.code == tc.EngineErrorCode["UseAfterShutdown"]
    assert rec.is_error() is True


def test_init_shutdown_init_rearms(assets):
    """init -> shutdown -> init on the same Engine handle must not segfault."""
    e = tc.Engine()
    _init_or_skip(e, assets)
    e.shutdown()
    # Re-init. The Vulkan loader may or may not allow a second VkInstance
    # in the same process; either outcome is acceptable.
    try:
        again = e.init(
            enable_validation=False,
            cache_dir=assets["cache_dir"],
            nodes_dir=assets["nodes_dir"],
            glsl_dir=assets["glsl_dir"],
        )
        if again:
            assert e.is_ready() is True
            e.shutdown()
        else:
            # Vulkan refused the re-init. Engine should be in the Error state.
            assert e.is_ready() is False
    except Exception:
        # An init exception is also acceptable -- we just need to not crash.
        pass


def engine_state_unwrap(e):
    """Helper for the new engine_state() accessor (returns the int value of
    Engine::EngineState)."""
    return int(e.engine_state())
