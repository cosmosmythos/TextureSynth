#!/usr/bin/env python3
"""Validate TextureSynth node manifests against their GLSL entry signatures."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NODES = ROOT / "shader_assets" / "nodes"
DEFAULT_GLSL = ROOT / "shader_assets" / "glsl"

REQUIRED_NODE_KEYS = {"id", "display_name", "inputs", "outputs", "params", "shader", "entry"}
ALLOWED_NODE_KEYS = REQUIRED_NODE_KEYS | {"description", "includes"}
ALLOWED_SOCKET_KEYS = {"name", "type", "description"}
ALLOWED_PARAM_KEYS = {
    "name",
    "display_name",
    "description",
    "default",
    "min",
    "max",
    "step",
    "integer",
    "as_socket",
    "internal",
}


def strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    source = re.sub(r"//.*?$", "", source, flags=re.M)
    return source


def split_args(arg_source: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(arg_source):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            arg = arg_source[start:i].strip()
            if arg:
                args.append(arg)
            start = i + 1
    tail = arg_source[start:].strip()
    if tail:
        args.append(tail)
    return args


def parse_signature(source: str, entry: str) -> list[tuple[str, str]]:
    clean = strip_comments(source)
    pattern = rf"\bvec4\s+{re.escape(entry)}\s*\((.*?)\)"
    match = re.search(pattern, clean, flags=re.S)
    if not match:
        raise ValueError(f"missing GLSL entry function vec4 {entry}(...)")

    parsed: list[tuple[str, str]] = []
    for arg in split_args(match.group(1)):
        parts = arg.split()
        if len(parts) < 2:
            raise ValueError(f"cannot parse argument '{arg}'")
        parsed.append((parts[-2], parts[-1]))
    return parsed


def check_keys(path: Path, obj: dict, allowed: set[str], where: str, errors: list[str]) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        errors.append(f"{path}: unknown {where} keys: {', '.join(unknown)}")


def validate_manifest(path: Path, nodes_dir: Path, glsl_dir: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"{path}: invalid JSON: {exc}"]

    missing = sorted(REQUIRED_NODE_KEYS - set(manifest))
    if missing:
        errors.append(f"{path}: missing required keys: {', '.join(missing)}")
        return errors

    check_keys(path, manifest, ALLOWED_NODE_KEYS, "node", errors)

    node_id = manifest["id"]
    shader_path = nodes_dir / manifest["shader"]
    if not shader_path.exists():
        errors.append(f"{path}: shader file not found: {shader_path}")
        return errors

    for inc in manifest.get("includes", []):
        inc_path = glsl_dir / f"{inc}.glsl"
        if not inc_path.exists():
            errors.append(f"{path}: include not found: {inc_path}")

    inputs = manifest.get("inputs", [])
    outputs = manifest.get("outputs", [])
    params = manifest.get("params", [])
    if not isinstance(inputs, list) or not isinstance(outputs, list) or not isinstance(params, list):
        errors.append(f"{path}: inputs, outputs, and params must be lists")
        return errors

    for i, sock in enumerate(inputs):
        check_keys(path, sock, ALLOWED_SOCKET_KEYS, f"input[{i}]", errors)
    for i, sock in enumerate(outputs):
        check_keys(path, sock, ALLOWED_SOCKET_KEYS, f"output[{i}]", errors)
    for i, param in enumerate(params):
        check_keys(path, param, ALLOWED_PARAM_KEYS, f"param[{i}]", errors)

    param_names = [p.get("name") for p in params]
    dup_params = sorted({name for name in param_names if name and param_names.count(name) > 1})
    if dup_params:
        errors.append(f"{path}: duplicate params in {node_id}: {', '.join(dup_params)}")

    try:
        signature = parse_signature(shader_path.read_text(encoding="utf-8"), manifest["entry"])
    except Exception as exc:
        errors.append(f"{path}: {exc}")
        return errors

    expected = [("vec2", "uv")]
    for inp in inputs:
        if inp.get("type") == "sampler2D":
            expected_type = "int"
            expected_name = f"{inp['name']}_slot"
        else:
            expected_type = "vec4"
            expected_name = inp["name"]
        expected.append((expected_type, expected_name))
    expected.extend(("float", param["name"]) for param in params)

    if len(signature) != len(expected):
        errors.append(
            f"{path}: {manifest['entry']} has {len(signature)} args, expected {len(expected)} "
            f"({', '.join(t + ' ' + n for t, n in expected)})"
        )
        return errors

    for index, ((got_type, got_name), (want_type, want_name)) in enumerate(zip(signature, expected)):
        if got_type != want_type or got_name != want_name:
            errors.append(
                f"{path}: arg {index} is '{got_type} {got_name}', "
                f"expected '{want_type} {want_name}'"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-dir", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--glsl-dir", type=Path, default=DEFAULT_GLSL)
    args = parser.parse_args()

    manifests = sorted(args.nodes_dir.glob("*.node.json"))
    if not manifests:
        print(f"No node manifests found in {args.nodes_dir}", file=sys.stderr)
        return 1

    errors: list[str] = []
    node_ids: dict[str, Path] = {}
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            node_id = manifest.get("id")
            if node_id in node_ids:
                errors.append(f"{path}: duplicate node id '{node_id}', first seen in {node_ids[node_id]}")
            elif node_id:
                node_ids[node_id] = path
        except Exception:
            pass
        errors.extend(validate_manifest(path, args.nodes_dir, args.glsl_dir))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Validated {len(manifests)} node manifests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
