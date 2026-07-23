"""
debug_pyd.py — run with Blender's Python to test texturesynth_core directly.
Usage:
  "C:\Program Files\Blender Foundation\Blender 5.0\5.0\python\bin\python.exe" debug_pyd.py

This bypasses all Blender addon machinery. If it crashes here, the problem
is in the pyd itself. If it works here but crashes in Blender, the problem
is in the addon Python code.
"""

import sys, os

# Point to where Blender installed the pyd
SITE_PACKAGES = os.path.join(
    os.environ["APPDATA"],
    r"Blender Foundation\Blender\5.0\extensions\.local\lib\python3.11\site-packages"
)
sys.path.insert(0, SITE_PACKAGES)

print(f"Python: {sys.version}")
print(f"Looking for pyd in: {SITE_PACKAGES}")
print()

# Step 1 — import
print("=== Step 1: import texturesynth_core ===")
try:
    import texturesynth_core as core
    print("  OK — module imported")
    print(f"  Module: {core}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Step 2 — construct Engine
print("\n=== Step 2: core.Engine() ===")
try:
    eng = core.Engine()
    print(f"  OK — Engine constructed: {eng}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Step 3 — init Vulkan
print("\n=== Step 3: eng.init() ===")
try:
    ok = eng.init()
    print(f"  Result: {ok}")
    if not ok:
        print(f"  Error: {eng.last_error()}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Step 4 — set a simple graph
print("\n=== Step 4: set_graph (Perlin -> Output) ===")
try:
    g = core.Graph()
    g.add_node(0, "perlin")
    g.set_output(0)
    ok = eng.set_graph(g)
    print(f"  Result: {ok}")
    if not ok:
        print(f"  Error: {eng.last_error()}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

# Step 5 — poll for compile
print("\n=== Step 5: poll until pipeline ready (max 5s) ===")
import time
for i in range(100):
    try:
        eng.poll_pending_compiles()
    except Exception as e:
        print(f"  poll FAILED: {e}")
        sys.exit(1)
    if eng.has_pipeline():
        print(f"  Pipeline ready after {i * 0.05:.2f}s")
        break
    time.sleep(0.05)
else:
    print(f"  Timed out. Last error: {eng.last_error()}")
    sys.exit(1)

# Step 6 — dispatch + readback
print("\n=== Step 6: render_dispatch_readback ===")
try:
    pc = core.PushConstants()
    pc.resolution_x = 512
    pc.resolution_y = 512
    pc.seed = 42
    pc.time = 0.0
    pixels = eng.render_dispatch_readback(pc)
    print(f"  OK — pixels shape: {pixels.shape}, dtype: {pixels.dtype}")
    print(f"  Sample pixel [256,256]: {pixels[256,256]}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\n=== ALL STEPS PASSED ===")
eng.shutdown()
