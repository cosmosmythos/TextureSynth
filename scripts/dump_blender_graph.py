"""Dump TextureSynth node tree: nodes, links, preview root, bake output.

Run inside Blender:
  1. Scripting workspace -> Text Editor -> Open this file -> Run Script
  2. Or paste into a new text block and Alt+P
  3. Output appears in the System Console (Window -> Toggle System Console)

Read-only — does not modify the graph or engine.
"""

import bpy


def _find_ts_tree():
    ctx = bpy.context
    space = getattr(ctx, "space_data", None)
    if space is not None and getattr(space, "type", None) == "NODE_EDITOR":
        for attr in ("edit_tree", "node_tree"):
            tree = getattr(space, attr, None)
            if tree is not None and tree.bl_idname == "TextureSynthTreeType":
                return tree

    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", None) == "TextureSynthTreeType":
            return ng
    return None


def _stable_id(node):
    if hasattr(node, "stable_id"):
        return node.stable_id()
    return hash(node.name) & 0xFFFFFFFFFFFFFFFF


def _socket_index(node, sock):
    sockets = node.outputs if sock.is_output else node.inputs
    return list(sockets).index(sock)


def dump_graph(tree=None):
    tree = tree or _find_ts_tree()
    if tree is None:
        print("No TextureSynthTreeType node group found.")
        return

    print("=" * 72)
    print(f"Tree: {tree.name!r}  nodes={len(tree.nodes)}  links={len(tree.links)}")
    print("=" * 72)

    name_to_id = {}
    for n in tree.nodes:
        if getattr(n, "sv_type", None) is not None or n.bl_idname == "TS_Output_Node":
            name_to_id[n.name] = _stable_id(n)

    active = getattr(tree.nodes, "active", None)
    output_node = None
    for n in tree.nodes:
        if n.bl_idname == "TS_Output_Node":
            output_node = n
            break

    print("\n--- NODES ---")
    for n in sorted(tree.nodes, key=lambda x: x.name):
        sid = name_to_id.get(n.name)
        flags = []
        if active is n:
            flags.append("ACTIVE")
        if getattr(n, "mute", False):
            flags.append("MUTED")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        sv = getattr(n, "sv_type", None)
        print(
            f"  {n.name!r}  id=0x{sid:016x}  type={n.bl_idname}  sv_type={sv!r}{flag_str}"
        )
        for i, s in enumerate(n.inputs):
            link_info = "unlinked"
            if s.is_linked and s.links:
                ln = s.links[0]
                if ln.is_valid and ln.from_node is not None:
                    fi = _socket_index(ln.from_node, ln.from_socket)
                    link_info = f"{ln.from_node.name!r}[out {fi}]"
            print(f"      in[{i}] {s.name!r} <- {link_info}")

    print("\n--- CONNECTIONS (engine-relevant) ---")
    conn_count = 0
    for link in tree.links:
        if not link.is_valid:
            continue
        src = link.from_node
        dst = link.to_node
        if src is None or dst is None:
            continue
        if src.name not in name_to_id or dst.name not in name_to_id:
            continue
        if output_node is not None and (
            src.name == output_node.name or dst.name == output_node.name
        ):
            continue
        si = _socket_index(src, link.from_socket)
        di = _socket_index(dst, link.to_socket)
        print(
            f"  {src.name!r}[out {si}] -> {dst.name!r}[in {di}]  "
            f"(ids 0x{name_to_id[src.name]:016x} -> 0x{name_to_id[dst.name]:016x})"
        )
        conn_count += 1
    if conn_count == 0:
        print("  (none)")

    if output_node is not None:
        print(f"\n--- OUTPUT / BAKE ({output_node.name!r}) ---")
        for i, s in enumerate(output_node.inputs):
            if s.is_linked and s.links:
                ln = s.links[0]
                src = ln.from_node
                si = _socket_index(src, ln.from_socket) if src else 0
                src_name = src.name if src else "<invalid>"
                print(f"  bake target {s.name!r} <- {src_name!r}[out {si}]")
            else:
                print(f"  bake target {s.name!r} <- <unlinked>")

    try:
        from bl_ext.user_default.texturesynth.core import engine_bridge
    except ImportError:
        try:
            import texturesynth.core.engine_bridge as engine_bridge
        except ImportError:
            engine_bridge = None

    if engine_bridge is not None:
        root, _, _, order, _ = engine_bridge._resolve_preview_root(tree)
        print("\n--- PREVIEW (TS_Preview2D driver) ---")
        if root is None:
            print("  preview root: <none>")
        else:
            rid = name_to_id.get(root.name)
            print(
                f"  preview root: {root.name!r}  id=0x{rid:016x}  "
                f"muted={bool(getattr(root, 'mute', False))}"
            )
            print(f"  upstream order ({len(order)}): {order}")

        graph, _, _ = engine_bridge._build_graph_and_params(tree)
        if graph is not None:
            print(f"  engine graph.output_node id: 0x{int(graph.output_node):016x}")
        else:
            print("  engine graph: <could not build>")

    print("\nDone.")


dump_graph()
