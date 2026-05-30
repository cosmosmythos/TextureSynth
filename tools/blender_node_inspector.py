"""
TextureSynth Node Architecture Inspector
=========================================
Run this in Blender's Text Editor or Python console to dump the
complete node architecture for reference.  It shows every registered
node class, its sockets, properties, categories, and color scheme.

Usage:
    1. Enable the TextureSynth addon in Blender Preferences.
    2. Open this file in Blender's Text Editor.
    3. Run Script (Alt+P).
    4. Check the System Console (Window → Toggle System Console).

The JSON manifest will be written to:
    C:/Users/User/Documents/0/TEXTURESYNTH/node_architecture_reference.json

To use the C++ library inspector, run this script from Blender WITH the
TextureSynth addon enabled.  If run standalone, the script falls back to
Blender-side data only (node classes registered without C++ are still found).
"""

import bpy
import json
import sys
import os

# Ensure the project root is on sys.path so we can import ADDON modules.
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Also try common build output locations for texturesynth_core.pyd.
_PYD_SEARCH = [
    os.path.join(_SCRIPT_DIR, "build", "Release"),
    os.path.join(_SCRIPT_DIR, "build"),
]
for p in _PYD_SEARCH:
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)

OUTPUT_PATH = r"C:\Users\User\Documents\0\TEXTURESYNTH\node_architecture_reference.json"


def inspect_all() -> dict:
    """Scrape every registered class and build a reference dictionary."""

    report = {
        "_meta": {
            "Blender Version": ".".join(str(v) for v in bpy.app.version),
            "Addon": "TextureSynth",
            "Tree Type": "TextureSynthTreeType",
        },
        "socket_types": {},
        "node_types": {},
        "node_tree": None,
        "categories": {},
        "nodes_in_scene": [],
        "engine_library": {},
    }

    # ── 1. Inspect TextureSynth tree class ──────────────────────────
    for cls in bpy.types.NodeTree.__subclasses__():
        if getattr(cls, "bl_idname", None) == "TextureSynthTreeType":
            report["node_tree"] = {
                "bl_idname": cls.bl_idname,
                "bl_label": getattr(cls, "bl_label", ""),
                "bl_icon": getattr(cls, "bl_icon", ""),
            }
            break

    # ── 2. Find TextureSynth node trees in current .blend ───────────
    for nt in bpy.data.node_groups:
        if getattr(nt, "bl_idname", None) == "TextureSynthTreeType":
            report["nodes_in_scene"].append({
                "tree_name": nt.name,
                "node_count": len(nt.nodes),
                "link_count": len(nt.links),
                "nodes": [
                    {
                        "name": n.name,
                        "bl_idname": n.bl_idname,
                        "sv_type": getattr(n, "sv_type", "<N/A>"),
                        "ts_category": getattr(n, "ts_category", ""),
                        "format_override": getattr(n, "format_override", "DEFAULT"),
                        "uuid": getattr(n, "ts_uuid", "")[:12] + "...",
                        "inputs": [s.bl_idname for s in n.inputs],
                        "outputs": [s.bl_idname for s in n.outputs],
                    }
                    for n in nt.nodes
                ],
            })

    # ── 3. Socket types used by TextureSynth ────────────────────────
    TREE_TYPES = {report["node_tree"]["bl_idname"]} if report["node_tree"] else set()
    for cls in bpy.types.NodeSocket.__subclasses__():
        bl_id = getattr(cls, "bl_idname", None)
        if bl_id and bl_id.startswith("TS_"):
            report["socket_types"][bl_id] = {
                "bl_label": getattr(cls, "bl_label", ""),
                "color": list(cls.draw_color_simple()) if hasattr(cls, "draw_color_simple") else None,
            }

    # ── 4. Node classes registered under TextureSynth tree ──────────
    registered_bl_ids = set()
    for cls in bpy.types.Node.__subclasses__():
        bl_id = getattr(cls, "bl_idname", None)
        if bl_id and bl_id.startswith("TS_"):
            registered_bl_ids.add(bl_id)

    # If no TS_ classes are registered, try forcing addon import to
    # trigger class registration (e.g. when run from Blender Text Editor
    # without the addon being loaded via Preferences).
    if not registered_bl_ids:
        try:
            import ADDON
            if hasattr(ADDON, 'register'):
                ADDON.register()
                print("[Inspector] ADDON.register() called — classes should now be registered.")
                # Re-scan
                for cls in bpy.types.Node.__subclasses__():
                    bl_id = getattr(cls, "bl_idname", None)
                    if bl_id and bl_id.startswith("TS_"):
                        registered_bl_ids.add(bl_id)
        except Exception as e:
            print(f"[Inspector] Could not auto-load ADDON: {e}")

    for cls in bpy.types.Node.__subclasses__():
        bl_id = getattr(cls, "bl_idname", None)
        if not bl_id or not bl_id.startswith("TS_"):
            continue

        # Gather annotations that are bpy.props (filters out methods)
        props = {}
        for k, v in getattr(cls, "__annotations__", {}).items():
            if hasattr(v, "keywords"):
                kw = v.keywords
                subtype = str(v)
                if "RNAPI" in subtype:
                    subtype = subtype.split(" ")[1] if " " in subtype else "Unknown"
                props[k] = {
                    "type": subtype.split(".")[-1].split("'")[0],
                    "default": kw.get("default"),
                    "min": kw.get("min"),
                    "max": kw.get("max"),
                }

        entry = {
            "bl_idname": bl_id,
            "bl_label": getattr(cls, "bl_label", ""),
            "sv_type": getattr(cls, "sv_type", None),
            "ts_category": getattr(cls, "ts_category", "INPUT"),
            "poll": getattr(cls, "poll", None).__doc__
            if callable(getattr(cls, "poll", None)) else "",
            "has_format_override": "format_override" in getattr(cls, "__annotations__", {}),
            "properties": props,
        }

        # Sample sockets from a fresh instance if possible
        try:
            nt = bpy.data.node_groups.get("__inspector_tmp__")
            if nt is None:
                nt = bpy.data.node_groups.new("__inspector_tmp__", "TextureSynthTreeType")
            node = nt.nodes.new(bl_id)
            entry["inputs"] = [
                {"name": s.name, "type": s.bl_idname}
                for s in node.inputs
            ]
            entry["outputs"] = [
                {"name": s.name, "type": s.bl_idname}
                for s in node.outputs
            ]
            nt.nodes.remove(node)
        except Exception as e:
            entry["_socket_error"] = str(e)

        report["node_types"][bl_id] = entry

    # Clean up temp tree
    tmp = bpy.data.node_groups.get("__inspector_tmp__")
    if tmp:
        bpy.data.node_groups.remove(tmp)

    # ── 5. Categories (from ts_category values across all nodes) ─────
    # These match TS_CATEGORY_COLORS in base.py
    report["categories"] = {
        "INPUT":  {"color": (0.20, 0.35, 0.55), "desc": "blue-ish — input sources"},
        "NOISE":  {"color": (0.25, 0.45, 0.30), "desc": "green-ish — procedural noise"},
        "COLOR":  {"color": (0.55, 0.40, 0.20), "desc": "orange-ish — color ops"},
        "FILTER": {"color": (0.45, 0.25, 0.45), "desc": "purple-ish — filters/effects"},
        "BLEND":  {"color": (0.55, 0.30, 0.30), "desc": "red-ish — blend modes"},
        "OUTPUT": {"color": (0.15, 0.15, 0.15), "desc": "neutral dark — output sink"},
    }

    # ── 6. Try to load C++ NodeLibrary for the full manifest ─────────
    try:
        from ADDON.core import cpp_module
        engine = cpp_module.get_engine()
        if engine:
            lib = engine.node_library()
            all_types = lib.all()
            for type_id, nt in all_types.items():
                entry = {
                    "display_name": nt.display_name,
                    "description": nt.description,
                    "inputs": [
                        {"name": s.name, "type": str(s.type), "format": str(s.format)}
                        for s in nt.inputs
                    ],
                    "outputs": [
                        {"name": s.name, "type": str(s.type), "format": str(s.format)}
                        for s in nt.outputs
                    ],
                    "params": [
                        {
                            "name": p.name,
                            "display_name": p.display_name,
                            "default": p.default_value,
                            "min": p.min_value,
                            "max": p.max_value,
                            "step": p.step,
                            "is_integer": p.is_integer,
                            "as_socket": p.as_socket,
                        }
                        for p in nt.params
                    ],
                    "variant_flags": list(nt.variant_flags),
                    "glsl_function": nt.glsl_function[:80] + "..."
                    if len(nt.glsl_function) > 80
                    else nt.glsl_function,
                }
                report["engine_library"][type_id] = entry

            print(f"[Inspector] C++ library loaded: {len(all_types)} node types")
        else:
            print("[Inspector] C++ engine not ready (no pipeline yet)")
    except Exception as e:
        print(f"[Inspector] C++ library unavailable: {e}")

    return report


def write_report(report: dict):
    """Write the report to a JSON file and print a summary to console."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"  Full reference written to: {OUTPUT_PATH}")
    print(f"{'='*70}")

    # ── Summary table ────────────────────────────────────────────────
    nt = report["node_tree"]
    if nt:
        print(f"\n  Tree type:      {nt['bl_idname']} ({nt['bl_label']})")
    print(f"  Sockets found:  {len(report['socket_types'])}")
    print(f"  Node classes:   {len(report['node_types'])}")
    print(f"  C++ manifests:  {len(report['engine_library'])}")

    print(f"\n  {'Node Class':<30} {'sv_type':<20} {'Category':<10} {'Inputs':<20} {'Outputs':<20}")
    print(f"  {'-'*30} {'-'*20} {'-'*10} {'-'*20} {'-'*20}")
    for bl_id, entry in sorted(report["node_types"].items()):
        inputs = ", ".join(s["type"].replace("TS_", "") for s in entry.get("inputs", []))
        outputs = ", ".join(s["type"].replace("TS_", "") for s in entry.get("outputs", []))
        print(f"  {bl_id:<30} {str(entry['sv_type']):<20} {entry['ts_category']:<10} {inputs:<20} {outputs:<20}")

    # ── Scene nodes ──────────────────────────────────────────────────
    for tree in report["nodes_in_scene"]:
        print(f"\n  Scene tree: '{tree['tree_name']}' — {tree['node_count']} nodes, {tree['link_count']} links")
        for n in tree["nodes"]:
            print(f"    {n['name']:<25} {n['bl_idname']:<30} fmt={n['format_override']:<8} cat={n['ts_category']}")

    print()


def print_node_graphviz(report: dict):
    """Print a Graphviz DOT digraph of all node types and their socket connections."""
    print("\n  # Copy-paste this into https://dreampuf.github.io/GraphvizOnline/")
    print("  digraph TextureSynthNodes {")
    print("    rankdir=LR;")
    print("    node [shape=box, style=rounded, fontname=monospace];")
    print('    ranksep=1.5;')
    cat_colors = {
        "INPUT": "#3399ff", "NOISE": "#33cc66", "COLOR": "#ff9933",
        "FILTER": "#9933cc", "BLEND": "#cc3333", "OUTPUT": "#333333",
    }
    for bl_id, entry in sorted(report["node_types"].items()):
        cat = entry.get("ts_category", "FILTER")
        color = cat_colors.get(cat, "#cccccc")
        label = f"{entry['bl_label']}\\n({entry['sv_type'] or 'output'})"
        # Socket annotations
        ins = "|".join(f"<i{i}> {s['name'] or '(none)'}" for i, s in enumerate(entry.get("inputs", [])))
        outs = "|".join(f"<o{i}> {s['name'] or '(none)'}" for i, s in enumerate(entry.get("outputs", [])))
        label = f"{{{ins}}}{{{label}}}{{{outs}}}"
        escaped_id = bl_id.replace("-", "_").replace(".", "_")
        print(f'    {escaped_id} [label="{label}", fillcolor="{color}", style="filled,rounded"];')
    print("  }")


if __name__ == "__main__":
    print(f"\n{'#'*70}")
    print(f"  TextureSynth Node Architecture Inspector")
    print(f"{'#'*70}\n")

    data = inspect_all()
    write_report(data)
    print_node_graphviz(data)

    print(f"\n{'#'*70}")
    print(f"  Done. Full JSON at: {OUTPUT_PATH}")
    print(f"{'#'*70}\n")
