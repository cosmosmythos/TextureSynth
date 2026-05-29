"""
Engine Bridge — the ONLY file that talks to texturesynth_core.

Node identity uses persistent UUID-derived uint64 IDs from TextureSynthNode.stable_id().
This makes graph identity independent of Blender's node ordering, node renaming,
and undo/redo proxy invalidation.
"""
import bpy
from . import cpp_module


_last_active_fingerprint = None
_submitted_generation = 0
_last_applied_generation = 0
_last_pushed_param_hash = None


def _find_node_tree():
    """Prefer the active TextureSynth node editor tree, fallback to a usable tree."""
    ctx = bpy.context

    space = getattr(ctx, "space_data", None)
    if space is not None and getattr(space, "type", None) == "NODE_EDITOR":
        for attr in ("edit_tree", "node_tree"):
            tree = getattr(space, attr, None)
            if tree is not None and getattr(tree, "bl_idname", None) == "TextureSynthTreeType":
                return tree

    candidates = [
        nt for nt in bpy.data.node_groups
        if getattr(nt, "bl_idname", None) == "TextureSynthTreeType"
    ]
    for nt in candidates:
        has_output = any(n.bl_idname == "TS_Output_Node" for n in nt.nodes)
        if has_output and len(nt.links) > 0:
            return nt
    return candidates[0] if candidates else None


def _node_stable_id(node):
    """Get the UUID-derived uint64 from a TextureSynth node.
    Falls back to hash(node.name) for nodes that don't have stable_id() (shouldn't happen)."""
    if hasattr(node, 'stable_id'):
        return node.stable_id()
    # Fallback: deterministic hash from name (for safety only)
    return hash(node.name) & 0xFFFFFFFFFFFFFFFF


def _build_id_maps(tree):
    """Build name<->stable_id maps. Keys are node.name strings, values are uint64 stable IDs."""
    name_to_id = {}
    id_to_name = {}
    for node in tree.nodes:
        sid = _node_stable_id(node)
        name_to_id[node.name] = sid
        id_to_name[sid] = node.name
    return name_to_id, id_to_name


def _node_by_name(tree, name):
    """Resolve a name back to a live node proxy. Returns None if gone."""
    return tree.nodes.get(name)


def _get_active_subgraph_topo_order(tree, output_node, name_to_id):
    """Topo sort of nodes reachable upstream from output_node.

    Returns (order_ids, reachable_ids_set). All identity via node.name -> stable_id.
    """
    reachable_names = set()

    def trace(node):
        if node is None:
            return
        if node.name in reachable_names:
            return
        reachable_names.add(node.name)
        for input_socket in node.inputs:
            if input_socket.is_linked:
                for link in input_socket.links:
                    if not link.is_valid:
                        continue
                    src = link.from_node
                    if src is None:
                        continue
                    if src.bl_idname == 'NodeUndefined':
                        continue
                    trace(src)

    trace(output_node)

    reachable_ids = {
        name_to_id[n] for n in reachable_names if n in name_to_id
    }

    in_degree = {nid: 0 for nid in reachable_ids}
    adj = {nid: [] for nid in reachable_ids}

    for link in tree.links:
        if not link.is_valid:
            continue
        src = link.from_node
        dst = link.to_node
        if src is None or dst is None:
            continue
        s_name = src.name
        d_name = dst.name
        if s_name not in name_to_id or d_name not in name_to_id:
            continue
        s_id = name_to_id[s_name]
        d_id = name_to_id[d_name]
        if s_id not in reachable_ids or d_id not in reachable_ids:
            continue
        adj[s_id].append(d_id)
        in_degree[d_id] += 1

    q = sorted([nid for nid, deg in in_degree.items() if deg == 0])
    order = []
    while q:
        cur = q.pop(0)
        order.append(cur)
        for nxt in sorted(adj[cur]):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                q.append(nxt)
                q.sort()

    return order, reachable_ids


def _build_graph_and_params(tree):
    core = cpp_module.get_core()
    if core is None:
        return None, None, None

    # ── Diagnostics ────────────────────────────────────────────────
    print(f"[TextureSynth] Tree '{tree.name}': "
          f"{len(tree.nodes)} nodes, {len(tree.links)} links")
    for n in tree.nodes:
        sid = _node_stable_id(n) if hasattr(n, 'stable_id') else '?'
        print(f"  node '{n.name}' bl_idname='{n.bl_idname}' "
              f"sv_type={getattr(n, 'sv_type', '<none>')} stable_id={sid}")
    for li, link in enumerate(tree.links):
        fn = link.from_node.name if link.from_node else "<none>"
        tn = link.to_node.name if link.to_node else "<none>"
        fb = link.from_node.bl_idname if link.from_node else "?"
        tb = link.to_node.bl_idname if link.to_node else "?"
        print(f"  link[{li}] {fn}({fb}) → {tn}({tb}) valid={link.is_valid}")
    # ──────────────────────────────────────────────────────────────

    graph = core.Graph()
    res = _get_resolution()
    render_res = _get_render_resolution(res)
    pc = core.PushConstants()
    pc.resolution_x = render_res
    pc.resolution_y = render_res
    pc.seed = 42
    pc.time = 0.0

    # Find Output node
    output_node = None
    for n in tree.nodes:
        if n.bl_idname == 'TS_Output_Node':
            output_node = n
            break
    if output_node is None:
        print("[TextureSynth] No Output node in tree.")
        return None, None, None

    name_to_id, id_to_name = _build_id_maps(tree)
    output_node_name = output_node.name

    # Find a valid link feeding the Output node
    output_src_id = None
    for link in tree.links:
        if not link.is_valid:
            continue
        if link.to_node is None or link.from_node is None:
            continue
        if link.to_node.name != output_node_name:
            continue
        src = link.from_node
        if src.bl_idname == 'NodeUndefined':
            print("[TextureSynth] Output is fed by NodeUndefined — "
                  "delete and recreate the source node.")
            continue
        if src.name in name_to_id:
            output_src_id = name_to_id[src.name]
            print(f"[TextureSynth] Output ← stable_id={output_src_id} "
                  f"name='{src.name}' bl_idname={src.bl_idname}")
            break

    if output_src_id is None:
        print("[TextureSynth] Output node has no valid incoming link.")
        return None, None, None

    order, reachable_ids = _get_active_subgraph_topo_order(
        tree, output_node, name_to_id)

    # Add nodes to graph (skip Output marker — it has no sv_type)
    for nid in order:
        nname = id_to_name.get(nid)
        if nname is None:
            continue
        node = _node_by_name(tree, nname)
        if node is None:
            continue
        if node.bl_idname == 'TS_Output_Node':
            continue        
        if getattr(node, 'sv_type', None) is None:
            continue
        graph.add_node(nid, node.sv_type)

    # Collect parameters in same topological order
    node_params = []
    for nid in order:
        nname = id_to_name.get(nid)
        if nname is None:
            continue
        node = _node_by_name(tree, nname)
        if node is None:
            continue
        if getattr(node, 'sv_type', None) is None:
            continue
        if hasattr(node, 'get_parameters'):
            params = node.get_parameters()
            node_params.extend(params)

    graph.set_output(output_src_id)

    # Add edges — skip Output marker
    for link in tree.links:
        if not link.is_valid:
            continue
        src = link.from_node
        dst = link.to_node
        if src is None or dst is None:
            continue
        s_name = src.name
        d_name = dst.name
        if s_name not in name_to_id or d_name not in name_to_id:
            continue
        if d_name == output_node_name or s_name == output_node_name:
            continue
        s_id = name_to_id[s_name]
        d_id = name_to_id[d_name]
        if s_id not in reachable_ids or d_id not in reachable_ids:
            continue
        try:
            src_idx = list(src.outputs).index(link.from_socket)
            dst_idx = list(dst.inputs).index(link.to_socket)
        except ValueError:
            continue
        graph.add_connection(s_id, src_idx, d_id, dst_idx)
        print(f"[TextureSynth] edge {s_id}.{src_idx} → {d_id}.{dst_idx}")

    return graph, pc, node_params


def _build_push_constants(tree):
    core = cpp_module.get_core()
    if core is None:
        return None, []

    res = _get_resolution()
    render_res = _get_render_resolution(res)
    pc = core.PushConstants()
    pc.resolution_x = render_res
    pc.resolution_y = render_res
    pc.seed = 42
    pc.time = 0.0

    name_to_id, id_to_name = _build_id_maps(tree)
    output_node = next((n for n in tree.nodes
                        if n.bl_idname == 'TS_Output_Node'), None)
    if output_node is None:
        return None, []

    order, _ = _get_active_subgraph_topo_order(tree, output_node, name_to_id)

    node_params = []
    for nid in order:
        nname = id_to_name.get(nid)
        if nname is None:
            continue
        node = _node_by_name(tree, nname)
        if node is None:
            continue
        if getattr(node, 'sv_type', None) is None:
            continue
        if hasattr(node, 'get_parameters'):
            params = node.get_parameters()
            print(f"[TextureSynth] params for {node.name} ({node.sv_type}): {params}")
            node_params.extend(params)

    return pc, node_params


def _get_resolution():
    try:
        return bpy.context.scene.texturesynth_resolution
    except (AttributeError, TypeError):
        return 512


def _get_render_resolution(res):
    try:
        scale_str = getattr(bpy.context.scene, "texturesynth_proxy_scale", '1.0')
        scale = float(scale_str)
        scaled = int(res * scale)
        return max(8, (scaled + 7) // 8 * 8)
    except Exception:
        return res


def _blit_to_image(pixels, width, height):
    img_name = "TS_Preview2D"
    img = bpy.data.images.get(img_name)
    if img is None:
        img = bpy.data.images.new(
            img_name, width=width, height=height,
            alpha=True, float_buffer=True,
        )
    elif img.size[0] != width or img.size[1] != height:
        img.scale(width, height)
    img.pixels.foreach_set(pixels.flatten())
    img.update()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


def _invalidate_output_image():
    """Clear the output image to black so artists see a failed render clearly."""
    import numpy as np
    img = bpy.data.images.get("TS_Preview2D")
    if img is None:
        return
    w, h = img.size
    if w == 0 or h == 0:
        return
    black = np.zeros(w * h * 4, dtype=np.float32)
    img.pixels.foreach_set(black)
    img.update()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


# ── Public API ──────────────────────────────────────────────────────

def _params_signature(tree):
    """Cheap hash of all active-subgraph slider values. Used to skip
    redundant async dispatches when nothing changed."""
    name_to_id, id_to_name = _build_id_maps(tree)
    output_node = next((n for n in tree.nodes
                        if n.bl_idname == 'TS_Output_Node'), None)
    if output_node is None:
        return None
    order, _ = _get_active_subgraph_topo_order(tree, output_node, name_to_id)
    parts = []
    for nid in order:
        nname = id_to_name.get(nid)
        if not nname:
            continue
        node = _node_by_name(tree, nname)
        if node is None or getattr(node, 'sv_type', None) is None:
            continue
        if hasattr(node, 'get_named_parameters'):
            try:
                kv = node.get_named_parameters()
                parts.append((nid, tuple(sorted(kv.items()))))
                continue
            except Exception:
                pass
        if hasattr(node, 'get_parameters'):
            parts.append((nid, tuple(node.get_parameters())))
    return tuple(parts)


def _active_subgraph_fingerprint(tree):
    """Stable hash of (active nodes + their types + active connections + output target).
    Uses UUID-based stable IDs, making the fingerprint independent of Blender node order."""
    output_node = next((n for n in tree.nodes
                        if n.bl_idname == 'TS_Output_Node'), None)
    if output_node is None:
        return ("no_output",)
    name_to_id, id_to_name = _build_id_maps(tree)
    order, reachable = _get_active_subgraph_topo_order(
        tree, output_node, name_to_id)
    # node identity: stable_id + sv_type
    node_part = tuple(
        (nid,
         getattr(_node_by_name(tree, id_to_name.get(nid, '')), 'sv_type', None))
        for nid in order
    )
    # active links only (using stable IDs)
    link_part = []
    for link in tree.links:
        if not link.is_valid:
            continue
        if link.from_node is None or link.to_node is None:
            continue
        sn, dn = link.from_node.name, link.to_node.name
        if sn not in name_to_id or dn not in name_to_id:
            continue
        sid, did = name_to_id[sn], name_to_id[dn]
        if sid not in reachable or did not in reachable:
            continue
        try:
            so = list(link.from_node.outputs).index(link.from_socket)
            di = list(link.to_node.inputs).index(link.to_socket)
        except ValueError:
            continue
        link_part.append((sid, so, did, di))
    link_part.sort()
    # output source
    out_src = None
    for link in tree.links:
        if link.is_valid and link.to_node and link.to_node.name == output_node.name:
            out_src = _node_stable_id(link.from_node) if link.from_node else None
            break
    return (node_part, tuple(link_part), out_src)


def upload_node_image(node):
    """Uploads the selected bpy.types.Image pixels to Vulkan via texturesynth_core."""
    engine = cpp_module.get_engine()
    if engine is None:
        return False
    
    nid = _node_stable_id(node)
    image = getattr(node, 'image', None)
    if image is None:
        engine.release_image(nid)
        return True
        
    w, h = image.size
    if w <= 0 or h <= 0:
        engine.release_image(nid)
        return True
        
    import numpy as np
    try:
        # Blender's image.pixels holds float RGBA values flatly.
        pixels = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        
        # Call the C++ engine to allocate staging buffer and upload to Vulkan image.
        success = engine.upload_image(nid, pixels, w, h)
        if success:
            print(f"[TextureSynth] Successfully uploaded external image '{image.name}' ({w}x{h}) for node stable_id={nid}")
        else:
            print(f"[TextureSynth] Core engine failed to upload image '{image.name}'")
        return success
    except Exception as e:
        print(f"[TextureSynth] Exception during image upload for node '{node.name}': {e}")
        engine.release_image(nid)
        return False


def submit_graph():
    global _last_active_fingerprint, _submitted_generation
    global _last_applied_generation, _last_pushed_param_hash
    engine = cpp_module.get_engine()
    if engine is None:
        return 0
    tree = _find_node_tree()
    if tree is None:
        return 0

    res = _get_resolution()
    engine.set_resolution(res, res)

    precision_str = getattr(bpy.context.scene, "texturesynth_precision", 'R16')
    if precision_str == 'R8':
        engine.set_precision(0)
    elif precision_str == 'R16':
        engine.set_precision(1)
    else:
        engine.set_precision(2)

    graph, pc, _ = _build_graph_and_params(tree)
    if graph is None:
        # Transient invalid state during editing (e.g. user dragging a link
        # away from a socket, Output node missing). DO NOT invalidate the
        # last successful render — the artist's viewport stays stable.
        print("[TextureSynth] No valid graph — Output node missing or disconnected.")
        _last_active_fingerprint = None
        _submitted_generation = 0
        return 0

    # Scan and pre-upload active Image Input nodes before setting the graph
    name_to_id, id_to_name = _build_id_maps(tree)
    output_node = next((n for n in tree.nodes if n.bl_idname == 'TS_Output_Node'), None)
    if output_node:
        order, _ = _get_active_subgraph_topo_order(tree, output_node, name_to_id)
        for nid in order:
            nname = id_to_name.get(nid)
            if nname:
                node = _node_by_name(tree, nname)
                if node and node.bl_idname == 'TS_Image_Node':
                    upload_node_image(node)

    try:
        generation = int(engine.set_graph(graph))
        if generation == 0:
            print(f"[TextureSynth] set_graph failed: {engine.last_error()}")
            sync_node_errors(tree)
            _invalidate_output_image()
            return 0
    except Exception as e:
        print(f"[TextureSynth] set_graph exception: {e}")
        return 0

    _last_active_fingerprint = _active_subgraph_fingerprint(tree)
    _submitted_generation = generation
    _last_applied_generation = 0
    _last_pushed_param_hash = None
    _push_params_using_stable_ids(tree, engine)
    _diagnose_params(engine)
    sync_node_errors(tree)
    print(f"[TextureSynth] submitted graph generation={generation}")
    return generation


def update_params_only(force_submit: bool = False):
    """Async render. Returns one of:
       'landed'      — fresh pixels were blitted this tick
       'in_flight'   — a job is queued/running, keep ticking
       'idle'        — nothing to do (no changes, ring quiet)
       'invalid'     — engine/tree/pipeline not ready
    Caller (evaluation.py) decides what to do with each state.
    """
    global _last_active_fingerprint, _last_applied_generation
    global _last_pushed_param_hash

    engine = cpp_module.get_engine()
    if engine is None or not engine.has_pipeline():
        return 'invalid'

    # ── FAST PATH: idle drain. No tree walk. No logs. No push.
    if not force_submit:
        return _poll_and_blit(engine)

    # ── SLOW PATH: dispatch a render.
    tree = _find_node_tree()
    if tree is None:
        return 'invalid'

    res = _get_resolution()
    engine.set_resolution(res, res)
    precision_str = getattr(bpy.context.scene, "texturesynth_precision", 'R16')
    engine.set_precision(0 if precision_str == 'R8'
                         else 1 if precision_str == 'R16' else 2)

    # Stale-pipeline detection.
    fp = _active_subgraph_fingerprint(tree)
    if fp != _last_active_fingerprint:
        print("[TextureSynth] active subgraph changed under param update — resubmitting.")
        from .evaluation import request_topology_update
        request_topology_update()
        return 'invalid'

    pc, _ = _build_push_constants(tree)
    if pc is None:
        return 'invalid'

    try:
        if _submitted_generation and not engine.is_generation_ready(_submitted_generation):
            return _poll_and_blit(engine)

        # Skip push+submit if params unchanged since last submit.
        sig = _params_signature(tree)
        if sig is not None and sig == _last_pushed_param_hash:
            return _poll_and_blit(engine)

        _push_params_using_stable_ids(tree, engine)
        _last_pushed_param_hash = sig

        ticket = engine.submit_render(pc, _submitted_generation)
        landed = _poll_and_blit(engine)
        if landed == 'landed':
            return 'landed'
        return 'in_flight' if ticket != 0 else 'idle'

    except Exception as e:
        print(f"[TextureSynth] async render exception: {e}")
        return 'invalid'


def _poll_and_blit(engine):
    """Drain at most one completed result. Drops stale gens. Returns
    'landed' if we blitted something new, else 'idle'."""
    global _last_applied_generation
    try:
        result = engine.poll_readback()
    except Exception as e:
        print(f"[TextureSynth] poll_readback exception: {e}")
        return 'idle'

    if result is None:
        return 'idle'

    pixels, gen = result

    # ── Stale-gen guard ─────────────────────────────────────────────
    # If C++ hands us a result older than what we've already applied,
    # drop it silently. (Shouldn't happen post-Fix-B in C++, but defensive.)
    if gen != 0 and gen < _last_applied_generation:
        print(f"[TextureSynth] dropping stale gen={gen} "
              f"(already applied gen={_last_applied_generation})")
        return 'idle'

    height, width = pixels.shape[:2]
    _blit_to_image(pixels, width, height)
    _last_applied_generation = gen
    print(f"[TextureSynth] async render landed gen={gen} "
          f"{width}x{height} min={pixels.min():.3f} "
          f"max={pixels.max():.3f} mean={pixels.mean():.3f}")
    return 'landed'


def submitted_generation():
    return _submitted_generation


def sync_node_errors(tree):
    """Polls the core engine for compile errors and updates node UI colors/tooltips."""
    engine = cpp_module.get_engine()
    if engine is None:
        return

    fid = engine.failed_node()
    last_err = engine.last_error()

    # If failed_node is 0, we should ensure all nodes are cleared of error states.
    # If failed_node is non-zero, we mark that specific node as failed (red color + stashed message),
    # and clear the error on all other nodes.
    from ..nodes.base import TS_CATEGORY_COLORS
    name_to_id, id_to_name = _build_id_maps(tree)
    failed_node_name = id_to_name.get(fid) if fid != 0 else None

    for node in tree.nodes:
        if getattr(node, 'sv_type', None) is None and node.bl_idname != 'TS_Output_Node':
            continue
        
        # Check if this node is the one that failed
        if failed_node_name and node.name == failed_node_name:
            # Set custom color to a vibrant red (0.8, 0.1, 0.1)
            node.use_custom_color = True
            node.color = (0.8, 0.1, 0.1)
            if hasattr(node, 'ts_compile_error'):
                node.ts_compile_error = last_err
        else:
            # Reset color back to its category color
            cat = getattr(node, 'ts_category', 'FILTER')
            color = TS_CATEGORY_COLORS.get(cat, (0.45, 0.25, 0.45))
            node.color = color
            node.use_custom_color = True
            if hasattr(node, 'ts_compile_error') and node.ts_compile_error:
                node.ts_compile_error = ""


def _push_params_using_stable_ids(tree, engine):
    """Send slider values keyed by UUID-derived stable IDs.

    Uses name-based binding (Phase 6) so that reordering parameters in a
    node's JSON manifest cannot silently shift slider values between
    unrelated params. The Python node must expose `get_named_parameters()`
    returning {param_name: float}; if it doesn't, we fall back to the
    positional API for backward compatibility.
    """
    layout = engine.param_layout()  # dict {stable_id:int -> base:int}
    for node in tree.nodes:
        if getattr(node, 'sv_type', None) is None:
            continue
        nid = _node_stable_id(node)
        if nid not in layout:
            continue  # node not in active subgraph

        # Preferred: name-based binding.
        if hasattr(node, 'get_named_parameters'):
            try:
                kv = node.get_named_parameters()
                if kv:
                    engine.update_node_params_by_name(nid, kv)
                    print(f"[TextureSynth] NAMED push '{node.name}': {kv}")
                    continue
            except Exception as e:
                print(f"[TextureSynth] get_named_parameters failed for "
                      f"'{node.name}': {e} — falling back to positional")

        # Fallback: positional (legacy). Should not happen post-Phase 6;
        # warn once per node so we notice contract regressions.
        if hasattr(node, 'get_parameters'):
            if not getattr(node, '_ts_warned_positional', False):
                print(f"[TextureSynth] WARN: '{node.name}' ({node.sv_type}) "
                      f"lacks get_named_parameters() — using positional API. "
                      f"Sliders are vulnerable to JSON param reordering.")
                try:
                    node['_ts_warned_positional'] = True
                except Exception:
                    pass
            params = node.get_parameters()
            engine.update_node_params_by_id(nid, params)


def _diagnose_params(engine):
    layout = engine.param_layout()
    total = engine.total_param_floats()
    print(f"[TextureSynth] param_layout={dict(layout)} total_floats={total}")
