"""Direct C++ graph build via engine API to capture logs."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mcp_fetch import send

code = r"""
import bpy, sys, time

result = {}

# Find modules
tsc = None; cpp_mod = None
for k, v in sys.modules.items():
    if k == "texturesynth_core": tsc = v
    elif k.endswith("core.cpp_module"): cpp_mod = v

if not all([tsc, cpp_mod]):
    result["error"] = "Missing modules"
    import json as j; print(j.dumps(result))
else:
    # Set up C++ logging BEFORE any engine call
    log_buf = []
    def log_cb(level, msg):
        log_buf.append(f"[{level}] {msg}")
    tsc.set_log_callback(log_cb)

    e = cpp_mod.get_engine()

    # Build a simple Perlin -> Blur -> Blend graph via C++ API
    try:
        g = tsc.Graph()
        perlin_id = 100
        levels_id = 101
        blur_id = 102
        blend_id = 103
        worley_id = 104
        g.add_node(perlin_id, "perlin", 0, "Perlin", False, False, 0, 0)
        g.add_node(levels_id, "levels", 0, "Levels", False, False, 0, 0)
        g.add_node(blur_id, "blur", 0, "Blur", False, False, 0, 0)
        g.add_node(blend_id, "blend", 0, "Blend", False, False, 0, 0)
        g.add_node(worley_id, "worley", 0, "Worley", False, False, 0, 0)
        g.add_connection(perlin_id, 0, levels_id, 0)
        g.add_connection(levels_id, 0, blur_id, 0)
        g.add_connection(blur_id, 0, blend_id, 2)  # blur -> blend A (socket 2)
        g.add_connection(worley_id, 0, blend_id, 1) # worley -> blend B (socket 1)
        g.set_output(blend_id)

        result["graph_built"] = True

        # Set resolution and submit
        e.set_resolution(128, 128)
        gen = e.set_graph(g)
        result["generation"] = gen

        if gen != 0:
            # Wait for compile
            time.sleep(1.0)
            ready = False
            for _ in range(50):
                e.poll_pending_compiles()
                if e.is_generation_ready(gen):
                    ready = True
                    break
                time.sleep(0.05)
            result["compile_ready"] = ready

            # Readback
            pc = tsc.PushConstants()
            pc.resolution_x = 128; pc.resolution_y = 128; pc.seed = 1; pc.time = 0.0
            ticket = e.submit_render(pc, gen)
            if ticket:
                time.sleep(0.5)
                arr = e.readback_sync()
                h, w, c = arr.shape
                result["readback"] = {
                    "shape": [h, w, c],
                    "sample": [round(float(arr[h//2, w//2, ci]), 6) for ci in range(min(c,4))]
                }
    except Exception as ex:
        result["build_error"] = str(ex)

    # Collect all log lines
    result["log_all"] = log_buf

    import json as j; print(j.dumps(result, indent=2, default=str))
"""

resp = send(code)
import json
print(json.dumps(resp, indent=2, default=str))
