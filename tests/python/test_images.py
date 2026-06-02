"""
Image upload + release.
"""
import numpy as np
import pytest
import texturesynth_core as tc


def test_upload_image_correct_shape_returns_true(engine):
    """Upload a 32x32 RGBA image. Engine should accept (shape, no error)."""
    pixels = np.zeros((32, 32, 4), dtype=np.float32)
    pixels[..., 0] = 1.0  # red
    ok = engine.upload_image(1, pixels, 32, 32)
    assert ok is True
    rec = engine.last_error_record()
    # On success the record should be cleared (set_error_ was not called).
    assert rec.code == tc.EngineErrorCode['None']


def test_upload_image_release_roundtrip(engine):
    """Upload then release the same node id. Upload is async — must poll."""
    import time
    pixels = np.zeros((16, 16, 4), dtype=np.float32)
    engine.upload_image(2, pixels, 16, 16)  # node 2 (1 is reserved for a graph)
    # Wait for the upload to complete + register in image_registry_
    ok = False
    for _ in range(40):
        engine.poll_pending_compiles()
        # Try to release; success means registry has the image
        if engine.release_image(2):
            ok = True
            break
        time.sleep(0.05)
    assert ok, f"release_image kept returning False (last error: {engine.last_error_record().message!r})"
    # After the successful release, clear the stale ImageReleaseUnknown from
    # the failed attempts so the final assertion is meaningful.
    engine.clear_error()
    rec = engine.last_error_record()
    assert rec.code == tc.EngineErrorCode['None']


def test_release_image_unknown_id_is_error(engine):
    """Releasing an image that was never uploaded must fail with ImageReleaseUnknown."""
    ok = engine.release_image(123456)
    assert ok is False
    rec = engine.last_error_record()
    assert rec.code == tc.EngineErrorCode.ImageReleaseUnknown
    assert rec.failed_node == 123456
    assert rec.phase == tc.EnginePhase.ImageRelease
