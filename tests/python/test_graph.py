"""
Graph construction + structured error reporting.
"""
import pytest
import texturesynth_core as tc


def test_set_graph_empty_returns_error(engine):
    """An empty graph (no nodes, no output) must fail with GraphValidation."""
    g = tc.Graph()
    gen = engine.set_graph(g)
    assert gen == 0, f"empty graph should return 0, got {gen}"
    rec = engine.last_error_record()
    assert rec.code == tc.EngineErrorCode.GraphValidation
    assert rec.phase == tc.EnginePhase.GraphSubmit
    assert rec.failed_node == 0
    assert rec.message  # non-empty


def test_set_graph_unknown_type_returns_error(engine):
    """add_node with bogus type: depends on whether Graph validates at add_node
    or set_graph. Currently the check is at set_graph via validate_graph."""
    g = tc.Graph()
    try:
        g.add_node(1, "definitely_not_a_real_node")
    except Exception:
        return  # add_node itself rejected -- fine
    gen = engine.set_graph(g)
    assert gen == 0
    rec = engine.last_error_record()
    assert rec.code in (
        tc.EngineErrorCode.GraphValidation,
        tc.EngineErrorCode.GraphCompile,
    )


def test_set_graph_valid_returns_nonzero_generation(engine):
    g = tc.Graph()
    g.add_node(1, "perlin")
    g.set_output(1)
    gen = engine.set_graph(g)
    assert gen != 0, f"valid graph should return non-zero gen, got {gen}"
    rec = engine.last_error_record()
    assert rec.code == tc.EngineErrorCode['None']


def test_clear_error_resets_state(engine):
    engine.last_error_record()  # just access to confirm no crash
    # Force an error
    engine.set_graph(tc.Graph())
    rec = engine.last_error_record()
    assert rec.code != tc.EngineErrorCode['None']
    # Clear and verify
    engine.clear_error()
    rec = engine.last_error_record()
    assert rec.code == tc.EngineErrorCode['None']
    assert not rec.is_error()
    assert rec.failed_node == 0
    assert rec.graph_generation == 0
    assert rec.phase == tc.EnginePhase.Idle


def test_engine_error_struct_has_expected_fields():
    """EngineError has 5 fields + is_error method."""
    e = tc.EngineError()
    assert e.code == tc.EngineErrorCode['None']
    assert e.message == ""
    assert e.failed_node == 0
    assert e.graph_generation == 0
    assert e.phase == tc.EnginePhase.Idle
    assert e.is_error() is False

    e.code = tc.EngineErrorCode.GraphValidation
    e.message = "test"
    e.failed_node = 42
    e.graph_generation = 7
    e.phase = tc.EnginePhase.GraphSubmit
    assert e.is_error() is True
    assert e.failed_node == 42


def test_add_node_accepts_debug_name(engine):
    """Phase 1d: Graph::add_node can carry a human-readable debug_name
    that surfaces in engine logs and error messages."""
    g = tc.Graph()
    g.add_node(1, "perlin", debug_name="TerrainBase")
    g.set_output(1)
    gen = engine.set_graph(g)
    assert gen != 0, f"set_graph failed: {engine.last_error_record().message!r}"
    # The internal IR now stores debug_name="TerrainBase"; C++ test asserts this.
    # On the Python side we just verify the graph round-trips without error.


def test_add_node_debug_name_optional(engine):
    """debug_name is optional; default value is "" (empty)."""
    g = tc.Graph()
    g.add_node(1, "perlin")                  # no debug_name kwarg
    g.add_node(2, "grayscale", debug_name="WithLabel")
    g.add_connection(1, 0, 2, 0)              # grayscale has 1 input
    g.set_output(2)
    gen = engine.set_graph(g)
    assert gen != 0, f"set_graph failed: {engine.last_error_record().message!r}"


def test_add_node_muted_bypassed_optional(engine):
    """Phase 1c: Graph::add_node accepts muted and bypassed kwargs. Defaults
    are False; supplying True must not crash set_graph (the IR rewire /
    clear-pass is exercised by the C++ test suite)."""
    g = tc.Graph()
    # Defaults
    g.add_node(1, "perlin")
    # Explicit False
    g.add_node(2, "perlin", muted=False, bypassed=False)
    # bypassed=True (engine: dispatch emits vkCmdClearColorImage)
    g.add_node(3, "perlin", bypassed=True)
    # muted=True (engine: validator rewires; node excluded from IR)
    g.add_node(4, "perlin", muted=True)
    # Both kwargs together
    g.add_node(5, "perlin", muted=True, bypassed=True)
    g.set_output(1)
    gen = engine.set_graph(g)
    assert gen != 0, (
        f"set_graph failed: code={engine.last_error_record().code} "
        f"message={engine.last_error_record().message!r}"
    )


def test_add_node_bypassed_does_not_error(engine):
    """A graph whose only non-output node is bypassed must still compile
    successfully: bypassed passes skip the shader compile (executor will
    clear their output to zero) but the plan itself is valid."""
    g = tc.Graph()
    g.add_node(1, "perlin", bypassed=True)
    g.set_output(1)
    gen = engine.set_graph(g)
    assert gen != 0, (
        f"bypassed-only graph failed: code={engine.last_error_record().code} "
        f"message={engine.last_error_record().message!r}"
    )
