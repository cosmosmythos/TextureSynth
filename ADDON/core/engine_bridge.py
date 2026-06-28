"""Bridge between the Blender addon and texturesynth_core. Graph identity uses stable_id()
so it survives Blender node reordering/invalidation."""
import json
import os

import bpy
import numpy as np

from . import cpp_module
from . import logging as _tslog


_last_active_fingerprint = None
_submitted_generation = 0
_last_applied_generation = 0
_last_pushed_param_hash = None
_last_active_node_id = None
_last_mute_snapshot = {}  # node.name -> bool(mute)

# B1: image-upload change-detection cache
# node_stable_id -> (cheap_sig, content_hash)
_image_cache = {}
# B3: ring-full retry queue — node_stable_id -> (image_ref, w, h)
_pending_image_uploads = {}

_HASH_SAMPLE_TARGET = 256  # pixels to sample for content hash


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


def poll_mute_state(tree):
    """Detect mute-state changes across all nodes. Returns True if any node was muted/unmuted."""
    global _last_mute_snapshot
    current = {}
    for n in tree.nodes:
        if getattr(n, 'sv_type', None) is not None or n.bl_idname == 'TS_Output_Node':
            current[n.name] = bool(getattr(n, 'mute', False))
    changed = current != _last_mute_snapshot
    _last_mute_snapshot = current
    return changed


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


def _resolve_preview_root(tree):
    """Pick the node whose upstream subgraph is the current preview.
    Output node never drives the preview — the active node always does."""
    name_to_id, id_to_name = _build_id_maps(tree)

    active = getattr(tree.nodes, "active", None)
    if (active is not None
            and getattr(active, "sv_type", None) is not None
            and active.name in name_to_id
            and active.bl_idname != 'TS_Output_Node'):
        root = active
    else:
        root = None
        for n in tree.nodes:
            if (getattr(n, "sv_type", None) is not None
                    and n.name in name_to_id
                    and n.bl_idname != 'TS_Output_Node'):
                root = n
                break
    if root is None:
        return None, name_to_id, id_to_name, [], set()
    order, reachable = _get_active_subgraph_topo_order(tree, root, name_to_id)
    return root, name_to_id, id_to_name, order, reachable


def _get_active_subgraph_topo_order(tree, output_node, name_to_id):
    """Topological sort order and reachable IDs upstream from output_node."""
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
                    if src is None or src.bl_idname == 'NodeUndefined':
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


def _get_node_as_socket_names(node):
    """Resolve as_socket param names for a node: instance attr, then class attr,
    then C++ node library, then JSON manifest."""
    names = getattr(node, '_as_socket_names', None)
    if names and isinstance(names, frozenset):
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
                nt = lib.all().get(sv)
                if nt is not None:
                    return frozenset(
                        p.name for p in nt.params
                        if getattr(p, 'as_socket', False)
                    )
            except Exception:
                pass
    try:
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


def _build_graph_and_params(tree):
    core = cpp_module.get_core()
    if core is None:
        return None, None, None

    graph = core.Graph()
    res = _get_resolution()
    pc = core.PushConstants()
    pc.resolution_x = res
    pc.resolution_y = res
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
            and not getattr(active, 'mute', False)
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
            if src.bl_idname == 'NodeUndefined' or getattr(src, 'mute', False):
                continue
            if src.name in name_to_id:
                output_src_id = name_to_id[src.name]
                break
    if output_src_id is None and root is not None and root.name in name_to_id:
        if not getattr(root, 'mute', False):
            output_src_id = name_to_id[root.name]

    # When root is muted, trace upstream through its inputs to find the first non-muted
    # source — mirrors C++ resolve_muted_source().
    if output_src_id is None and root is not None:
        visited = set()
        queue = [root]
        while queue and output_src_id is None:
            cur = queue.pop(0)
            if cur.name in visited:
                continue
            visited.add(cur.name)
            for inp in cur.inputs:
                if not inp.is_linked:
                    continue
                for link in inp.links:
                    if not link.is_valid:
                        continue
                    src = link.from_node
                    if src is None or src.name in visited:
                        continue
                    if getattr(src, 'mute', False):
                        queue.append(src)
                        continue
                    if src.name in name_to_id and name_to_id[src.name] in reachable_ids:
                        output_src_id = name_to_id[src.name]
                        break

    if output_src_id is None:
        return None, None, None

    # Add ALL nodes (not just the active subgraph) so set_active_node can switch to
    # any node without recompiling the graph.
    for nname, nid in name_to_id.items():
        node = _node_by_name(tree, nname)
        if node is None:
            continue
        if node.bl_idname == 'TS_Output_Node':
            continue
        if getattr(node, 'sv_type', None) is None:
            continue
        fmt_override = node.get_format_override() if hasattr(node, 'get_format_override') else None
        depth_mode = node.get_depth_mode() if hasattr(node, 'get_depth_mode') else None
        abs_depth = node.get_absolute_depth() if hasattr(node, 'get_absolute_depth') else None
        depth_kwargs = {}
        if depth_mode is not None:
            depth_kwargs['depth_mode'] = depth_mode
        if abs_depth is not None:
            depth_kwargs['absolute_depth'] = abs_depth

        graph.add_node(
            nid, node.sv_type, fmt_override, node.name,
            muted=bool(getattr(node, 'mute', False)),
            bypassed=False,
            **depth_kwargs,
        )

    node_params = []
    for nname, nid in name_to_id.items():
        node = _node_by_name(tree, nname)
        if node is None:
            continue
        if getattr(node, 'sv_type', None) is None:
            continue
        if hasattr(node, 'get_parameters'):
            node_params.extend(node.get_parameters())

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
        graph.add_output_target(int(src.stable_id()), sock.name or "Unnamed", src_idx)

    # Connections between nodes. dst_idx maps Blender's socket order onto the C++ layout:
    # regular inputs come first ([0..reg_count)), as_socket params follow ([reg_count..)).
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
        try:
            src_idx = list(src.outputs).index(link.from_socket)
        except ValueError:
            continue
        as_names = _get_node_as_socket_names(dst)
        dst_idx = 0
        for sock in dst.inputs:
            if sock == link.to_socket:
                break
            if sock.name not in as_names:
                dst_idx += 1
        if link.to_socket.name in as_names:
            reg_count = sum(1 for s in dst.inputs if s.name not in as_names)
            dst_inputs = list(dst.inputs)
            as_before = sum(1 for s in dst_inputs
                            if s.name in as_names
                            and dst_inputs.index(s) < dst_inputs.index(link.to_socket))
            dst_idx = reg_count + as_before
        graph.add_connection(s_id, src_idx, d_id, dst_idx)

    return graph, pc, node_params


def _build_push_constants(tree):
    core = cpp_module.get_core()
    if core is None:
        return None, []

    res = _get_resolution()
    pc = core.PushConstants()
    pc.resolution_x = res
    pc.resolution_y = res
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
        if node is None or getattr(node, 'sv_type', None) is None:
            continue
        if hasattr(node, 'get_parameters'):
            node_params.extend(node.get_parameters())

    return pc, node_params


def _get_resolution():
    try:
        return bpy.context.scene.texturesynth_resolution
    except (AttributeError, TypeError):
        return 1024


def _redraw_image_editors():
    """Tag every IMAGE_EDITOR area for redraw after a pixel blit."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()


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
    _redraw_image_editors()


def _invalidate_output_image():
    """Clear the output image so a failed render is visible."""
    img = bpy.data.images.get("TS_Preview2D")
    if img is None:
        return
    w, h = img.size
    if w == 0 or h == 0:
        return
    black = np.zeros(w * h * 4, dtype=np.float32)
    img.pixels.foreach_set(black)
    img.update()
    _redraw_image_editors()


def _params_signature(tree):
    """Lightweight hash signature of all active parameters in the graph."""
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
        content = node.get_content_signature()
        if hasattr(node, 'get_named_parameters'):
            try:
                kv = node.get_named_parameters()
                if kv is not None:
                    parts.append((nid, tuple(sorted(kv.items())), content))
                    continue
            except Exception:
                pass
        if hasattr(node, 'get_parameters'):
            parts.append((nid, tuple(node.get_parameters()), content))
    return tuple(parts)


def _active_subgraph_fingerprint(tree):
    """Stable topology fingerprint of active nodes, connections, and output source."""
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


def _image_content_hash(image):
    """Strided sample of image pixels → bytes hash. Catches save/reload that reset is_dirty."""
    w, h = image.size
    if w <= 0 or h <= 0:
        return b""
    # Read full pixels once, sample rows 0, mid, last
    all_pixels = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(all_pixels)
    rows = {0, h // 2, h - 1}
    samples = []
    for r in sorted(rows):
        start = r * w * 4
        count = min(w, 32) * 4  # first 32 pixels per row
        samples.append(all_pixels[start:start + count].tobytes())
    return b"".join(samples)


def upload_node_image(node):
    """Upload the node's selected bpy.types.Image pixels to Vulkan via texturesynth_core.
    Skips re-upload if the image hasn't changed (B1 cache)."""
    engine = cpp_module.get_engine()
    if engine is None:
        return False

    nid = _node_stable_id(node)
    image = getattr(node, 'image', None)
    if image is None:
        engine.release_image(nid)
        _image_cache.pop(nid, None)
        return True

    w, h = image.size
    if w <= 0 or h <= 0:
        engine.release_image(nid)
        _image_cache.pop(nid, None)
        return True

    # Cheap signature: pointer, dims, source type, dirty flag
    sig = (image.as_pointer(), w, h, image.source, bool(image.is_dirty))
    cached = _image_cache.get(nid)
    if cached is not None:
        old_sig, old_hash = cached
        if sig == old_sig:
            return True  # no change at all
        if sig[:4] == old_sig[:4] and sig[4] != old_sig[4]:
            # is_dirty flipped (edit saved) — need content hash to confirm
            ch = _image_content_hash(image)
            if ch == old_hash:
                _image_cache[nid] = (sig, ch)
                return True  # pixels unchanged

    # Upload
    try:
        pixels = np.empty(w * h * 4, dtype=np.float32)
        image.pixels.foreach_get(pixels)
        pixels = pixels.reshape((h, w, 4))
        ok = engine.upload_image(nid, pixels, w, h)
        if ok:
            _image_cache[nid] = (sig, _image_content_hash(image))
        else:
            # B3: ring full — queue for retry next tick
            _pending_image_uploads[nid] = (image, w, h)
        return ok
    except Exception as e:
        _tslog.error(f"Exception during image upload for node '{node.name}': {e}")
        engine.release_image(nid)
        _image_cache.pop(nid, None)
        return False


def _apply_sidebar_precision(engine):
    """Read sidebar 'Precision' and set graph default depth on the engine."""
    precision_str = getattr(bpy.context.scene, "texturesynth_precision", 'R16')
    depth_map = {'R8': 0, 'R16': 1, 'R32': 2}
    legacy_mode = depth_map.get(precision_str, 1)
    if hasattr(engine, 'set_graph_default_depth'):
        try:
            from . import cpp_module
            BitDepth = cpp_module.BitDepth
            bd = {0: BitDepth.F8, 1: BitDepth.F16, 2: BitDepth.F32}.get(legacy_mode, BitDepth.F16)
            engine.set_graph_default_depth(bd)
            return
        except Exception:
            pass
    engine.set_precision(legacy_mode)


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

    _apply_sidebar_precision(engine)

    graph, pc, _ = _build_graph_and_params(tree)
    if graph is None:
        # Transient invalid state (e.g. missing Output node).
        _last_active_fingerprint = None
        _submitted_generation = 0
        return 0

    # DIAG: dump ALL Blender tree links (valid + invalid)
    for link in tree.links:
        mark = "OK" if link.is_valid else "INVALID"
        print(f"  [{mark}] {link.from_node.name}.{link.from_socket.name} -> {link.to_node.name}.{link.to_socket.name}")

    # B3: drain pending ring-full retries from previous ticks.
    if _pending_image_uploads:
        for nid, (img, rw, rh) in list(_pending_image_uploads.items()):
            if img is None or not img.name:
                _pending_image_uploads.pop(nid, None)
                continue
            cw, ch = img.size
            if cw != rw or ch != rh:
                _pending_image_uploads.pop(nid, None)
                continue
            pixels = np.empty(cw * ch * 4, dtype=np.float32)
            img.pixels.foreach_get(pixels)
            pixels = pixels.reshape((ch, cw, 4))
            if engine.upload_image(nid, pixels, cw, ch):
                _pending_image_uploads.pop(nid, None)
                sig = (img.as_pointer(), cw, ch, img.source, bool(img.is_dirty))
                _image_cache[nid] = (sig, _image_content_hash(img))

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

    print(f"[DIAG] submit_graph: gen={generation} output_node={graph.output_node}")
    _last_active_fingerprint = _active_subgraph_fingerprint(tree)
    _submitted_generation = generation
    _last_applied_generation = 0
    _last_pushed_param_hash = None

    _push_params_using_stable_ids(tree, engine)
    return generation


def update_params_only(force_submit: bool = False):
    """Trigger async render dispatch. Returns 'landed', 'in_flight', 'idle', or 'invalid'."""
    global _last_active_fingerprint, _last_applied_generation
    global _last_pushed_param_hash

    engine = cpp_module.get_engine()
    if engine is None or not engine.has_pipeline():
        return 'invalid'

    # Fast path: poll readback queue without checking/submitting params.
    if not force_submit:
        return _poll_and_blit(engine)

    tree = _find_node_tree()
    if tree is None:
        return 'invalid'

    # Active-node change = topology change; force full re-submit.
    if _check_active_node_change(tree, engine):
        from .evaluation import request_topology_update
        request_topology_update()
        return 'invalid'

    res = _get_resolution()
    engine.set_resolution(res, res)
    _apply_sidebar_precision(engine)

    fp = _active_subgraph_fingerprint(tree)
    if fp != _last_active_fingerprint:
        print(f"[DIAG] update_params_only: fingerprint mismatch, requesting topology update")
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

        print(f"[DIAG] update_params_only: pushing params (sig changed)")
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
    global _last_active_node_id, _last_pushed_param_hash, _submitted_generation, _last_active_fingerprint
    if not _submitted_generation:
        return False
    active = getattr(tree.nodes, "active", None)
    if active is None or not hasattr(active, "stable_id"):
        return False
    if getattr(active, 'sv_type', None) is None:
        return False
    new_id = active.stable_id()
    if new_id == _last_active_node_id:
        return False
    print(f"[DIAG] _check_active_node_change: new_id={new_id} prev_id={_last_active_node_id}")
    try:
        gen = int(engine.set_active_node(new_id))
    except Exception as e:
        _tslog.error(f"set_active_node({new_id}) failed: {e}")
        return False
    if not gen:
        print(f"[DIAG] _check_active_node_change: set_active_node returned 0 (node not in graph?)")
        return False
    _last_active_node_id = new_id
    _submitted_generation = gen
    _last_pushed_param_hash = None
    _last_active_fingerprint = _active_subgraph_fingerprint(tree)
    print(f"[DIAG] _check_active_node_change: set_active_node gen={gen} OK")
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
    """Poll the core engine for compile errors and update node UI colors/tooltips."""
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

        if failed_node_name and node.name == failed_node_name:
            if not getattr(node, 'use_custom_color', False):
                node.use_custom_color = True
            node.color = (0.8, 0.1, 0.1)
            if hasattr(node, 'ts_compile_error') and node.ts_compile_error != last_err:
                node.ts_compile_error = last_err
        else:
            if getattr(node, 'use_custom_color', False):
                node.use_custom_color = False
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
            engine.update_node_params_by_id(nid, node.get_parameters())
