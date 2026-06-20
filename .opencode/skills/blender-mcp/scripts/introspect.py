"""READ-ONLY introspection of whatever TextureSynth graph is open right now.

Mode A entry point. Does NOT create, delete, link, or change anything.
Prints a structured report: trees found, every node (sv_type, params,
links), the Output targets, engine state, and the last error.

Always run this BEFORE proposing any test that might mutate the user's graph.
The output is the basis for the mandatory quiz (see SKILL.md "Mode A").

Send via Blender MCP execute_code. Output comes back as captured stdout.
"""
import sys
import importlib


def _ts_pkg():
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        if name in sys.modules:
            return importlib.import_module(name)
    for mod in list(sys.modules.values()):
        core = getattr(mod, "core", None)
        if core is not None and hasattr(core, "cpp_module"):
            return mod
    for name in ("bl_ext.user_default.texturesynth", "texturesynth"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise RuntimeError("TextureSynth addon not loaded")


def main():
    import bpy
    ts = _ts_pkg()
    core = ts.core
    engine_bridge = core.engine_bridge
    cpp_module = core.cpp_module

    # --- All TextureSynth trees in the file ---
    ts_trees = [ng for ng in bpy.data.node_groups
                if getattr(ng, "bl_idname", None) == "TextureSynthTreeType"]
    print(f"=== TextureSynth trees: {len(ts_trees)} ===")
    for t in ts_trees:
        print(f"  - {t.name!r}  nodes={len(t.nodes)} links={len(t.links)}")

    active_tree = engine_bridge._find_node_tree()
    active_name = repr(active_tree.name) if active_tree else "None"
    print(f"\nactive tree (engine_bridge._find_node_tree): {active_name}")

    # --- Engine state ---
    engine = cpp_module.get_engine()
    print(f"\n=== engine ===")
    print(f"  loaded : {cpp_module.is_loaded()}")
    print(f"  engine : {engine is not None}")
    if engine is not None:
        gen = engine_bridge._submitted_generation
        try:
            ready = engine.is_generation_ready(gen) if gen else False
        except Exception as e:
            ready = f"<err {e!r}>"
        try:
            inflight = engine.async_in_flight()
        except Exception:
            inflight = "?"
        print(f"  submitted_generation : {gen}")
        print(f"  generation ready     : {ready}")
        print(f"  async_in_flight      : {inflight}")
        try:
            rec = engine.last_error_record()
            print(f"  last_error code      : {rec.code} phase={rec.phase}")
            print(f"  last_error message   : {rec.message!r}")
            print(f"  failed_node id       : {engine.failed_node()}")
        except Exception as e:
            print(f"  last_error unreadable: {e!r}")

    if active_tree is None:
        print("\n(no active TS tree -- nothing else to introspect)")
        return

    # --- Node inventory with params + link wiring ---
    name_to_id = {}
    for n in active_tree.nodes:
        if hasattr(n, "stable_id"):
            name_to_id[n.name] = n.stable_id()

    print(f"\n=== nodes in {active_tree.name!r}: {len(active_tree.nodes)} ===")
    for n in active_tree.nodes:
        sv = getattr(n, "sv_type", None)
        sid = name_to_id.get(n.name)
        mute = getattr(n, "mute", False)
        active = active_tree.nodes.active is n
        flags = " ".join(f for f, on in (
            ("ACTIVE", active), ("MUTED", bool(mute))) if on)
        print(f"\n  [{n.bl_idname}] name={n.name!r} sv_type={sv!r} "
              f"id={sid} {('('+flags+')') if flags else ''}")
        # Params: anything in __annotations__ that isn't a socket.
        ann = getattr(type(n), '__annotations__', {})
        param_vals = {}
        for key in ann:
            if key in ("ts_uuid", "ts_compile_error", "format_override",
                       "ts_category", "_ts_output_meta", "_as_socket_names"):
                continue
            try:
                param_vals[key] = getattr(n, key)
            except Exception:
                pass
        if param_vals:
            print(f"    params: {param_vals}")
        err = getattr(n, "ts_compile_error", "")
        if err:
            print(f"    COMPILE ERROR: {err!r}")
        # Input sockets + whether linked, and from whom.
        for i, s in enumerate(n.inputs):
            src = ""
            if s.is_linked and s.links:
                ln = s.links[0]
                src = (f" <- {ln.from_node.name!r}"
                       f"[{list(ln.from_node.outputs).index(ln.from_socket)}]")
            print(f"    in [{i}] {s.name!r} ({s.bl_idname}){src}")
        for i, s in enumerate(n.outputs):
            dsts = []
            for ln in active_tree.links:
                if ln.is_valid and ln.from_node is n and ln.from_socket is s:
                    dsts.append(ln.to_node.name)
            dst = f" -> {dsts!r}" if dsts else ""
            print(f"    out[{i}] {s.name!r} ({s.bl_idname}){dst}")

    # --- Output node target summary (the bake sink) ---
    for n in active_tree.nodes:
        if n.bl_idname == 'TS_Output_Node':
            print(f"\n=== Output node {n.name!r} targets ===")
            for i, s in enumerate(n.inputs):
                src = (s.links[0].from_node.name if s.is_linked and s.links
                       else "<unlinked>")
                print(f"    [{i}] {s.name!r} <- {src}")

    # --- Preview image ---
    img = bpy.data.images.get("TS_Preview2D")
    if img is not None:
        w, h = img.size
        print(f"\n=== preview image TS_Preview2D: {w}x{h} ===")
        if w and h:
            try:
                import numpy as np
                pix = np.empty(w * h * 4, dtype=np.float32)
                img.pixels.foreach_get(pix)
                print(f"    mean={float(pix.mean()):.4f} "
                      f"std={float(pix.std()):.4f} "
                      f"nonzero={float((pix != 0).mean()):.3f}")
            except Exception as e:
                print(f"    pixel stats unreadable: {e!r}")
    else:
        print("\n=== preview image TS_Preview2D: <not created yet> ===")


main()
