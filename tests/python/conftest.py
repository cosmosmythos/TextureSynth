"""
Shared pytest fixtures for texturesynth_core binding tests.

Adds `build/Release` to sys.path so `import texturesynth_core` works without
installing the wheel. Skips everything if Vulkan init fails (no GPU/driver).
"""
import os
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
BUILD_RELEASE = REPO_ROOT / "build" / "Release"

# Make `texturesynth_core` importable.
if str(BUILD_RELEASE) not in sys.path:
    sys.path.insert(0, str(BUILD_RELEASE))

# Stable asset paths (relative to repo root).
NODES_DIR = REPO_ROOT / "shader_assets" / "nodes"
GLSL_DIR  = REPO_ROOT / "shader_assets" / "glsl"
# Vulkan shader cache used by the Python tests. Lives INSIDE tests/ so it
# can be ignored by tests/.gitignore without polluting the root.
CACHE_DIR = TESTS_DIR / "python" / "cache" / "shader"

import texturesynth_core as tc  # noqa: E402


@pytest.fixture(scope="session")
def assets():
    """Resolved paths to shader assets + cache dir."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "nodes_dir":  str(NODES_DIR),
        "glsl_dir":   str(GLSL_DIR),
        "cache_dir":  str(CACHE_DIR),
    }


@pytest.fixture(scope="session")
def engine(assets):
    """
    A live Engine instance. Initialized once per session.
    Skips the entire test session if Vulkan init fails.
    """
    eng = tc.Engine()
    ok = eng.init(
        enable_validation=False,
        cache_dir=assets["cache_dir"],
        nodes_dir=assets["nodes_dir"],
        glsl_dir=assets["glsl_dir"],
    )
    if not ok:
        pytest.skip(
            f"Vulkan init failed: code={eng.last_error_record().code} "
            f"msg={eng.last_error_record().message!r}"
        )
    yield eng
    eng.shutdown()
