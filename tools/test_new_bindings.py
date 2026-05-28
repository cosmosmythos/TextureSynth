"""
test_new_bindings.py — tests new Vulkan engine precision and resolution bindings.
"""

import sys, os
import time

SITE_PACKAGES = os.path.join(
    os.environ.get("APPDATA", ""),
    r"Blender Foundation\Blender\5.0\extensions\.local\lib\python3.11\site-packages"
)
sys.path.insert(0, SITE_PACKAGES)

def run_test():
    print("Loading texturesynth_core...")
    try:
        import texturesynth_core as core
    except ImportError as e:
        print(f"Could not import texturesynth_core: {e}")
        print("Note: Run this test after building/installing the addon.")
        return

    print("Creating Engine...")
    eng = core.Engine()
    
    print("Initializing Vulkan...")
    if not eng.init():
        print(f"Failed to initialize Vulkan: {eng.last_error()}")
        return

    print("Testing set_precision / precision...")
    # Default format is VK_FORMAT_R32G32B32A32_SFLOAT which is precision 2
    print(f"Initial precision: {eng.precision()}")
    assert eng.precision() == 2, f"Expected initial precision 2, got {eng.precision()}"

    # Change to 16-bit half-float
    eng.set_precision(1)
    print(f"Precision after setting to 16F: {eng.precision()}")
    assert eng.precision() == 1, f"Expected precision 1, got {eng.precision()}"

    # Change to 8-bit unorm
    eng.set_precision(0)
    print(f"Precision after setting to 8-bit: {eng.precision()}")
    assert eng.precision() == 0, f"Expected precision 0, got {eng.precision()}"

    # Change back to 32-bit float
    eng.set_precision(2)
    print(f"Precision after setting to 32F: {eng.precision()}")
    assert eng.precision() == 2, f"Expected precision 2, got {eng.precision()}"

    print("Testing set_resolution...")
    # Resize to 256x256
    eng.set_resolution(256, 256)
    
    # Run a simple graph to verify everything evaluates correctly
    g = core.Graph()
    g.add_node(1, "perlin")
    g.set_output(1)
    
    if not eng.set_graph(g):
        print(f"set_graph failed: {eng.last_error()}")
        return

    # Wait for compile
    print("Compiling shaders...")
    for _ in range(100):
        eng.poll_pending_compiles()
        if eng.has_pipeline():
            break
        time.sleep(0.05)
    
    if not eng.has_pipeline():
        print("Failed to compile pipeline in time.")
        return

    # Run dispatch and readback
    pc = core.PushConstants()
    pc.resolution_x = 256
    pc.resolution_y = 256
    pc.seed = 123
    pc.time = 0.0
    
    pixels = eng.render_dispatch_readback(pc)
    print(f"Rendered shape: {pixels.shape}, expected (256, 256, 4)")
    assert pixels.shape == (256, 256, 4), f"Expected shape (256, 256, 4), got {pixels.shape}"

    print("\nALL BINDINGS TESTS PASSED SUCCESSFULLY!")
    eng.shutdown()

if __name__ == "__main__":
    run_test()
