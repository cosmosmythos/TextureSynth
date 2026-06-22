"""
Fused-chain cache-key collision regression test.

Reproduces the user-reported Blender symptom:

  Two graphs that share the same set of node types but wire them into blend's
  A/B sockets in swapped order must produce DIFFERENT GPU output. If they
  produce identical output, the FusedVariantKey is colliding: the second
  graph is being served the first graph's cached SPIR-V.

Why this specific shape (not in test_claim_verification.py):
  test_blend_mask_reaches_gpu swaps the MASK PARAMETER (no topology change).
  This test swaps the CONNECTION WIRING (same producer types, different
  sockets) -- the case the FusedVariantKey was missing until the
  internal_producer_indices field was added. build_fused_key() now packs
  node_type_ids + param_socket_masks + input_counts + feature_flags
  + external_socket_masks + internal_producer_indices (per-socket producer
  local_index from FusedGraphEmitter's RegSrc). epoch=8 invalidates every
  prior cache entry, so the swapped-wiring graphs get distinct keys.

Setup:
  Graph 1:  color_white -> blend.a , color_gray -> blend.b   (mode=mix, mask=1)
            output = blend -> mix(white, gray, 1.0) = gray
  Graph 2:  color_gray  -> blend.a , color_white -> blend.b  (mode=mix, mask=1)
            output = blend -> mix(gray, white, 1.0) = white

  Producers are different INSTANCES but same TYPE (color_const), so:
    - node_type_ids       identical: ["color_const", "color_const", "blend"]
    - param_socket_masks  identical
    - input_counts        identical
    - feature_flags       identical
    - external_socket_masks identical
    - internal_producer_indices DIFFERENT (this is what now breaks the tie)
  => distinct FusedVariantKey => no cache collision.

  Correct behavior: different output (gray vs white).

If this test PASSES, the key fix is working. If it FAILS with diff ~= 0, the
collision still exists and the cache is serving stale SPIR-V.
"""
import time
import numpy as np
import pytest
import texturesynth_core as tc


def _wait_pipeline(engine, ms=5000):
    for _ in range(ms // 50):
        engine.poll_pending_compiles()
        if engine.has_pipeline():
            return True
        time.sleep(0.05)
    return False


def _readback(engine, gen, ms=5000):
    pc = tc.PushConstants()
    pc.resolution_x = 64
    pc.resolution_y = 64
    pc.seed = 1
    pc.time = 0.0
    ticket = engine.submit_render(pc, gen)
    if ticket == 0:
        return None
    for _ in range(ms // 50):
        r = engine.poll_readback()
        if r is not None:
            arr, _g = r
            return arr
        time.sleep(0.05)
    return None


# color_const params: [mode=1 (flat), r, g, b, a]
WHITE = [1.0, 1.0, 1.0, 1.0, 1.0]
GRAY  = [1.0, 0.3, 0.3, 0.3, 1.0]


def _build_swap_graph(white_first):
    """
    Two color_const producers feeding blend.a (socket 1) and blend.b (socket 2).
    white_first=True  -> white on a, gray on b.
    white_first=False -> gray on a, white on b (swapped wiring).
    Both producers are distinct node IDs but same type (color_const).
    """
    g = tc.Graph()
    g.add_node(10, "color_const")
    g.add_node(11, "color_const")
    g.add_node(12, "blend")
    if white_first:
        g.add_connection(10, 0, 12, 1)   # node10 -> blend.a
        g.add_connection(11, 0, 12, 2)   # node11 -> blend.b
    else:
        g.add_connection(11, 0, 12, 1)   # node11 -> blend.a  (swapped)
        g.add_connection(10, 0, 12, 2)   # node10 -> blend.b  (swapped)
    g.set_output(12)
    return g


def test_swapped_internal_wiring_produces_different_output(engine):
    """Same producer types, swapped internal A/B wiring -> different GPU output."""
    # Graph 1: white->a, gray->b, mask=1 -> output = gray
    g1 = _build_swap_graph(white_first=True)
    gen1 = engine.set_graph(g1)
    assert gen1 != 0, f"set_graph(g1) failed: {engine.last_error()!r}"
    engine.update_node_params_by_id(10, WHITE)
    engine.update_node_params_by_id(11, GRAY)
    # blend params: [mode=0 (mix), mask=1.0] -> output = b
    engine.update_node_params_by_id(12, [0.0, 1.0])
    assert _wait_pipeline(engine), "g1 pipeline not ready"
    px1 = _readback(engine, engine.installed_generation())
    assert px1 is not None, "g1 readback failed"
    mean1 = float(np.mean(px1[..., :3]))

    # Graph 2: gray->a, white->b, mask=1 -> output = white (DIFFERENT wiring)
    g2 = _build_swap_graph(white_first=False)
    gen2 = engine.set_graph(g2)
    assert gen2 != 0, f"set_graph(g2) failed: {engine.last_error()!r}"
    engine.update_node_params_by_id(10, WHITE)
    engine.update_node_params_by_id(11, GRAY)
    engine.update_node_params_by_id(12, [0.0, 1.0])
    assert _wait_pipeline(engine), "g2 pipeline not ready"
    px2 = _readback(engine, engine.installed_generation())
    assert px2 is not None, "g2 readback failed"
    mean2 = float(np.mean(px2[..., :3]))

    diff = float(np.sum(np.abs(px1 - px2)))
    print(f"\n  g1 (white->a, gray->b, mask=1) mean: {mean1:.4f}  (expect ~0.3 = gray)")
    print(f"  g2 (gray->a, white->b, mask=1) mean: {mean2:.4f}  (expect ~1.0 = white)")
    print(f"  pixel diff: {diff:.4f}")

    # The two wirings should produce clearly different output. mask=1 makes
    # output = b, so g1 -> gray (~0.3) and g2 -> white (~1.0). A diff near 0
    # means the cache served g1's SPIR-V for g2 -- the collision bug.
    assert diff > 100.0, (
        f"Swapped internal wiring produced near-identical output (diff={diff:.4f}). "
        f"g1 mean={mean1:.4f}, g2 mean={mean2:.4f}. "
        "FusedVariantKey collides: build_fused_key's internal_producer_indices "
        "field is missing or not populated. Graph 2 was served Graph 1's "
        "cached SPIR-V."
    )
