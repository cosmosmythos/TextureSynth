"""
format_nodes.py — Pretty-print TextureSynth *.node.json files.

Rules:
  • Top-level keys on their own lines, 4-space indent.
  • Arrays of primitives (strings/numbers) inline on one line.
  • Arrays of objects: each object on ONE line, indented.
  • Preserves key order from the source file.
"""
import json
import sys
from pathlib import Path
from collections import OrderedDict


def fmt_scalar(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, float):
        # keep trailing .0 so ints-as-floats stay floats
        if v.is_integer():
            return f"{v:.1f}"
        return repr(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    raise TypeError(f"unsupported scalar: {type(v)}")


def fmt_inline_object(obj):
    """One-line { k: v, k: v } — used for array-of-object entries."""
    parts = [f'"{k}": {fmt_value_inline(v)}' for k, v in obj.items()]
    return "{ " + ", ".join(parts) + " }"


def fmt_value_inline(v):
    if isinstance(v, dict):
        return fmt_inline_object(v)
    if isinstance(v, list):
        return "[" + ", ".join(fmt_value_inline(x) for x in v) + "]"
    return fmt_scalar(v)


def fmt_array(key, arr, indent):
    """Decide between inline primitives array vs one-object-per-line."""
    pad = " " * indent
    if not arr:
        return "[]"
    # All primitives? → inline
    if all(not isinstance(x, (dict, list)) for x in arr):
        return "[" + ", ".join(fmt_scalar(x) for x in arr) + "]"
    # All objects → multi-line, one per line
    if all(isinstance(x, dict) for x in arr):
        inner_pad = " " * (indent + 4)
        lines = [fmt_inline_object(x) for x in arr]
        body = (",\n" + inner_pad).join(lines)
        return "[\n" + inner_pad + body + "\n" + pad + "]"
    # Mixed: fall back to JSON dump
    return json.dumps(arr, ensure_ascii=False)


def fmt_top_level(data):
    lines = ["{"]
    keys = list(data.keys())
    for i, k in enumerate(keys):
        v = data[k]
        if isinstance(v, list):
            rendered = fmt_array(k, v, indent=4)
        elif isinstance(v, dict):
            rendered = fmt_inline_object(v)
        else:
            rendered = fmt_scalar(v)
        comma = "," if i < len(keys) - 1 else ""
        lines.append(f'    "{k}": {rendered}{comma}')
    lines.append("}")
    return "\n".join(lines) + "\n"


def process_file(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f, object_pairs_hook=OrderedDict)
    except Exception as e:
        print(f"[skip] {path.name}: parse error: {e}")
        return False
    out = fmt_top_level(data)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"[ok]   {path.name}")
    return True


def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    if not target.is_dir():
        print(f"Not a directory: {target}")
        sys.exit(1)
    files = sorted(target.glob("*.json"))
    if not files:
        print(f"No .json files in {target}")
        return
    print(f"Formatting {len(files)} file(s) in {target}\n")
    ok = 0
    for p in files:
        if process_file(p):
            ok += 1
    print(f"\nDone: {ok}/{len(files)} formatted.")


if __name__ == "__main__":
    main()