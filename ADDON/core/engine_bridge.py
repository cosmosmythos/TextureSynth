"""Engine Bridge: provides interface between Blender addon and texturesynth_core C++ module.
Uses stable_id() to make graph identity independent of Blender node ordering or invalidation."""
import bpy
from . import cpp_module
from . import logging as _tslog


_last_active_fingerprint = None
_submitted_generation = 0
_last_applied_generation = 0
_last_pushed_param_hash = None
_last_active_node_id = None


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
    return candidates[0] if candidates else None


def _resolve_preview_root(tree):
    """Pick the node whose upstream subgraph is the current preview.
    Returns (root, name_to_id, id_to_name, topo_order, reachable_ids).
    """
    name_to_id, id_to_name = _build_id_maps(tree)

    # Check if Output node has any connected inputs (is it being *used*?)
    output_node = None
    output_has_inputs = False
    for n in tree.nodes:
        if n.bl_idname == 'TS_Output_Node':
            output_node = n
            for sock in n.inputs:
                if sock.is_linked:
                    output_has_inputs = True
                    break
            break

    # When Output exists but has no connections, ignore it and let
    # the active node (or any TS node) drive the preview.
    if output_node is None or not output_has_inputs:
        active = getattr(tree.nodes, "active", None)
        if (active is not None
                and getattr(active, "sv_type", None) is not None
                and active.name in name_to_id):
            root = active
        else:
            # If Output has no connections, fall back to active or any TS node.
            root = output_node if output_node else None
            if root is None:
                for n in tree.nodes:
                    if getattr(n, "sv_type", None) is not None and n.name in name_to_id:
                        root = n
                        break
    else:
        root = output_node
    if root is None:
        return None, name_to_id, id_to_name, [], set()
    order, reachable = _get_active_subgraph_topo_order(tree, root, name_to_id)
    return root, name_to_id, id_to_name, order, reachable


def _node_stable_id(node):
    """Get the UUID-derived uint64 from a TextureSynth node, falling back to name hash."""
    if hasattr(node, 'stable_id'):
        return node.stable_id()
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
    """Get topological sort order and reachable IDs upstream from output_node."""
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

    graph = core.Graph()
    res = _get_resolution()
    render_res = _get_render_resolution(res)
    pc = core.PushConstants()
    pc.resolution_x = render_res
    pc.resolution_y = render_res
    pc.seed = 42
    pc.time = 0.0

    root, name_to_id, id_to_name, order, reachable_ids = _resolve_preview_root(tree)
    if root is None or not order:
        return None, None, None

    output_node = root if root.bl_idname == 'TS_Output_Node' else None

    output_src_id = None
    active = getattr(tree.nodes, "active", None)
    if (output_node is not None
            and active is not None
            and hasattr(active, "stable_id")
            and getattr(active, "sv_type", None) is not None
            and active is not output_node
            and active.name in name_to_id
            and name_to_id[active.name] in reachable_ids):
        output_src_id = name_to_id[active.name]
    if output_src_id is None and output_node is not None:
        for link in tree.links:
            if not link.is_valid:
                continue
            if link.to_node is None or link.from_node is None:
                continue
            if link.to_node.name != output_node.name:
                continue
            src = link.from_node
            if src.bl_idname == 'NodeUndefined':
                continue
            if src.name in name_to_id:
                output_src_id = name_to_id[src.name]
                break
    if output_src_id is None and root is not None and root.name in name_to_id:
        output_src_id = name_to_id[root.name]

    if output_src_id is None:
        return None, None, None

    # Add nodes to graph, skipping Output node which is a marker.
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
        fmt_override = node.get_format_override() if hasattr(node, 'get_format_override') else None
        
        # Pass name to engine for logging. Mute rewires connections in the engine.
        is_muted = bool(getattr(node, 'mute', False))
        graph.add_node(
            nid, node.sv_type, fmt_override, node.name,
            muted=is_muted,
            bypassed=False,
        )

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

    # Map linked Output node input sockets to graph output targets.
    for sock in (output_node.inputs if output_node else []):
        if not sock.is_linked or not sock.links:
            continue
        src = sock.links[0].from_node
        if src is None or not hasattr(src, "stable_id"):
            continue
        from_sock = sock.links[0].from_socket
        src_idx = list(src.outputs).index(from_sock) if from_sock else 0
        graph.add_output_target(int(src.stable_id()), sock.name or "Unnamed",
                                src_idx)

    # Add connections between nodes.
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
        if output_node is not None and (d_name == output_node.name or s_name == output_node.name):
            continue
        s_id = name_to_id[s_name]
        d_id = name_to_id[d_name]
        if s_id not in reachable_ids or d_id not in reachable_ids:
            continue
        try:
            src_idx = list(src.outputs).index(link.from_socket)
        except ValueError:
            continue
        # Compute dst_idx: as_socket params occupy slots [inputs_n, inputs_n+count)
        # matching C++ GraphCompiler ordering. Blender's socket order may differ.
        as_names = _get_node_as_socket_names(dst)
        dst_idx = 0
        for sock in dst.inputs:
            if sock == link.to_socket:
                break
            if sock.name not in as_names:
                dst_idx += 1  # regular input → sequential slot
        # as_socket slots go AFTER all regular inputs
        if link.to_socket.name in as_names:
            reg_count = sum(1 for s in dst.inputs if s.name not in as_names)
            as_before = sum(1 for s in dst.inputs if s.name in as_names and
                            list(dst.inputs).index(s) < list(dst.inputs).index(link.to_socket))
            dst_idx = reg_count + as_before
        graph.add_connection(s_id, src_idx, d_id, dst_idx)

    return graph, pc, node_params


def _get_node_as_socket_names(node):
    """Resolve as_socket param names for a node. Checks instance attr first,
    then class attr, then falls back to C++ node library or JSON manifest."""
    names = getattr(node, '_as_socket_names', None)
    if names and isinstance(names, frozenset) and names:
        return names
    sv = getattr(node, 'sv_type', None)
    if sv is None:
        return frozenset()
    core = cpp_module.get_core()
    if core is not None:
        engine = cpp_module.get_engine()
        if engine is not None:
            try:
                lib = engine.node_library()
                all_types = lib.all()
                nt = all_types.get(sv)
                if nt is not None:
                    return frozenset(
                        p.name for p in nt.params
                        if getattr(p, 'as_socket', False)
                    )
            except Exception:
                pass
    import os, json
    try:
        import __main__
        addon_root = os.path.dirname(os.path.dirname(__file__))
        manifest_path = os.path.join(addon_root, 'shader_assets', 'nodes', f'{sv}.node.json')
        if os.path.isfile(manifest_path):
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return frozenset(
                p['name'] for p in data.get('params', [])
                if p.get('as_socket', False)
            )
    except Exception:
        pass
    return frozenset()


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

    root, name_to_id, id_to_name, order, _ = _resolve_preview_root(tree)
    if root is None or not order:
        return None, []

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

    return pc, node_params


def _get_resolution():
    try:
        return bpy.context.scene.texturesynth_resolution
    except (AttributeError, TypeError):
        return 1024


def _get_proxy_scale():
    """Get proxy scale divisor from scene property. 1 = full resolution."""
    try:
        scale_str = getattr(bpy.context.scene, "texturesynth_proxy_scale", '1.0')
        scale = float(scale_str)
        if scale <= 0:
            return 1
        # Convert float scale (e.g. 0.125) to divisor (e.g. 8)
        divisor = int(round(1.0 / scale))
        return max(1, divisor)
    except Exception:
        return 1


def _get_render_resolution(res):
    return res  # Phase 2: resolution is always full; proxy scale handled by engine dispatch


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
    """Clear the output image to see a failed render clearly."""
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


def _params_signature(tree):
    """Get a lightweight hash signature of all active parameters in the graph."""
    root, name_to_id, id_to_name, order, _ = _resolve_preview_root(tree)
    if root is None or not order:
        return None
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
    """Get stable topology fingerprint of active nodes, connections, and output source."""
    root, name_to_id, id_to_name, order, reachable = _resolve_preview_root(tree)
    if root is None or not order:
        return ("no_root",)

    node_part = tuple(
        (nid,
         getattr(_node_by_name(tree, id_to_name.get(nid, '')), 'sv_type', None),
         bool(getattr(_node_by_name(tree, id_to_name.get(nid, '')), 'mute', False)))
        for nid in order
    )

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

    if root.bl_idname == 'TS_Output_Node':
        out_src = None
        for link in tree.links:
            if link.is_valid and link.to_node and link.to_node.name == root.name:
                out_src = _node_stable_id(link.from_node) if link.from_node else None
                break
    else:
        out_src = _node_stable_id(root) if root else None
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
        # Extract flat float RGBA pixels from Blender image.
        pixels = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        
        # Upload pixels to Vulkan engine.
        success = engine.upload_image(nid, pixels, w, h)
        return success
    except Exception as e:
        _tslog.error(f"Exception during image upload for node '{node.name}': {e}")
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
    engine.set_proxy_scale(_get_proxy_scale())

    precision_str = getattr(bpy.context.scene, "texturesynth_precision", 'R16')
    if precision_str == 'R8':
        engine.set_precision(0)
    elif precision_str == 'R16':
        engine.set_precision(1)
    else:
        engine.set_precision(2)

    graph, pc, _ = _build_graph_and_params(tree)
    if graph is None:
        # Return early on transient invalid state (e.g., missing Output node).
        _last_active_fingerprint = None
        _submitted_generation = 0
        return 0

    # Pre-upload images from active image nodes.
    root, name_to_id, id_to_name, order, _ = _resolve_preview_root(tree)
    if root and order:
        for nid in order:
            nname = id_to_name.get(nid)
            if nname:
                node = _node_by_name(tree, nname)
                if node and node.bl_idname == 'TS_Image_Node':
                    upload_node_image(node)

    try:
        generation = int(engine.set_graph(graph))
        if generation == 0:
            _tslog.error(f"set_graph failed: {engine.last_error()}")
            sync_node_errors(tree)
            _invalidate_output_image()
            return 0
    except Exception as e:
        _tslog.error(f"set_graph exception: {e}")
        return 0

    _last_active_fingerprint = _active_subgraph_fingerprint(tree)
    _submitted_generation = generation
    _last_applied_generation = 0
    _last_pushed_param_hash = None

    _push_params_using_stable_ids(tree, engine)
    sync_node_errors(tree)
    return generation


def update_params_only(force_submit: bool = False):
    """Trigger async render dispatch and return status: landed, in_flight, idle, or invalid."""
    global _last_active_fingerprint, _last_applied_generation
    global _last_pushed_param_hash

    engine = cpp_module.get_engine()
    if engine is None or not engine.has_pipeline():
        return 'invalid'

    # Fast path: poll readback queue without checking/submitting params.
    if not force_submit:
        return _poll_and_blit(engine)

    # Slow path: check active node and submit render request.
    tree = _find_node_tree()
    if tree is None:
        return 'invalid'

    # Active change = topology change; force full re-submit.
    if _check_active_node_change(tree, engine):
        from .evaluation import request_topology_update
        request_topology_update()
        return 'invalid'

    res = _get_resolution()
    engine.set_resolution(res, res)
    engine.set_proxy_scale(_get_proxy_scale())
    precision_str = getattr(bpy.context.scene, "texturesynth_precision", 'R16')
    engine.set_precision(0 if precision_str == 'R8'
                         else 1 if precision_str == 'R16' else 2)

    fp = _active_subgraph_fingerprint(tree)
    if fp != _last_active_fingerprint:
        from .evaluation import request_topology_update
        request_topology_update()
        return 'invalid'

    pc, _ = _build_push_constants(tree)
    if pc is None:
        return 'invalid'

    try:
        if _submitted_generation and not engine.is_generation_ready(_submitted_generation):
            return _poll_and_blit(engine)

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
        _tslog.error(f"async render exception: {e}")
        return 'invalid'


def _check_active_node_change(tree, engine):
    """Update preview target in engine if selected node changed. Returns True on update."""
    global _last_active_node_id, _last_pushed_param_hash, _submitted_generation
    if not _submitted_generation:
        return False
    active = getattr(tree.nodes, "active", None)
    if active is None:
        return False
    if not hasattr(active, "stable_id"):
        return False
    if getattr(active, 'sv_type', None) is None:
        return False
    new_id = active.stable_id()
    if new_id == _last_active_node_id:
        return False
    try:
        gen = int(engine.set_active_node(new_id))
    except Exception as e:
        _tslog.error(f"set_active_node({new_id}) failed: {e}")
        return False
    if not gen:
        return False
    _last_active_node_id = new_id
    _submitted_generation = gen
    _last_pushed_param_hash = None
    _last_active_fingerprint = _active_subgraph_fingerprint(tree)
    return True


def _poll_and_blit(engine):
    """Poll engine for finished frames, blit to image, and return status."""
    global _last_applied_generation
    try:
        result = engine.poll_readback()
    except Exception as e:
        _tslog.error(f"poll_readback exception: {e}")
        return 'idle'

    if result is None:
        return 'idle'

    pixels, gen = result

    # Stale generation guard: ignore old frames.
    if gen != 0 and gen < _last_applied_generation:
        return 'idle'

    height, width = pixels.shape[:2]
    _blit_to_image(pixels, width, height)
    _last_applied_generation = gen
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

    name_to_id, id_to_name = _build_id_maps(tree)
    failed_node_name = id_to_name.get(fid) if fid != 0 else None

    for node in tree.nodes:
        if getattr(node, 'sv_type', None) is None and node.bl_idname != 'TS_Output_Node':
            continue
        
        # Reset colors for all nodes and highlight compile failure on the failed node.
        if failed_node_name and node.name == failed_node_name:
            node.use_custom_color = True
            node.color = (0.8, 0.1, 0.1)
            if hasattr(node, 'ts_compile_error'):
                node.ts_compile_error = last_err
        else:
            cat = getattr(node, 'ts_category', 'FILTER')
            if hasattr(node, 'ts_compile_error') and node.ts_compile_error:
                node.ts_compile_error = ""


def _push_params_using_stable_ids(tree, engine):
    """Send node parameters to engine using name-based or positional binding."""
    layout = engine.param_layout()
    for node in tree.nodes:
        if getattr(node, 'sv_type', None) is None:
            continue
        nid = _node_stable_id(node)
        if nid not in layout:
            continue

        if hasattr(node, 'get_named_parameters'):
            try:
                kv = node.get_named_parameters()
                if kv is not None:
                    engine.update_node_params_by_name(nid, kv)
                    continue
            except Exception:
                pass

        if hasattr(node, 'get_parameters'):
            params = node.get_parameters()
            engine.update_node_params_by_id(nid, params)
