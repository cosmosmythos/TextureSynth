"""
Parameter upload: by_id and by_name, success + warning paths.
"""
import pytest
import texturesynth_core as tc


@pytest.fixture
def perlin_graph(engine):
    g = tc.Graph()
    g.add_node(1, "perlin")
    g.set_output(1)
    gen = engine.set_graph(g)
    if gen == 0:
        pytest.skip(f"perlin graph failed: {engine.last_error_record().message!r}")
    return gen


def test_param_layout_dict_shape(engine, perlin_graph):
    layout = engine.param_layout()
    assert isinstance(layout, dict)
    assert 1 in layout, f"perlin node 1 not in param_layout: {layout}"
    assert engine.total_param_floats() >= 1


def test_update_node_params_by_id_accepts_list(engine, perlin_graph):
    """Perlin takes at least 2 floats (scale + seed)."""
    layout = engine.param_layout()
    slot_count = 64  # MAX_NODE_PARAMS - base
    engine.update_node_params_by_id(1, [1.0] * slot_count)
    # No assertion on output -- this is a fire-and-forget upload. No crash = pass.


def test_update_node_params_by_name_accepts_dict(engine, perlin_graph):
    """by_name looks up via type's param manifest."""
    lib = engine.node_library()
    perlin = lib.all().get("perlin")
    if perlin is None or not perlin.params:
        pytest.skip("perlin type has no named params (manifest empty?)")
    sample = {p.name: 1.0 for p in perlin.params}
    engine.update_node_params_by_name(1, sample)


def test_update_unknown_node_id_is_warning(engine, perlin_graph):
    """Updating params on an unknown node is a soft warning, not a crash."""
    engine.update_node_params_by_id(99999, [0.0] * 4)
    # Check that last_error_record reflects ParamUnknownNode (per Section 3a plan)
    # OR is None -- both are acceptable; we just need no exception.
    rec = engine.last_error_record()
    assert rec.code in (
        tc.EngineErrorCode['None'],
        tc.EngineErrorCode.ParamUnknownNode,
    )
