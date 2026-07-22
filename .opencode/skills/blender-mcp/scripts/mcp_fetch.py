"""MCP helper: send Python code to Blender's MCP server, return parsed JSON.

Usage:
    python mcp_fetch.py <code_string_or_file>
    python mcp_fetch.py "import bpy; result={'ver': bpy.app.version_string}"

Protocol: null-byte-delimited JSON over TCP to localhost:9876.
Response format: {"status": "ok"|"error", "result": {...}, "message": "..."}
"""

import json, socket, sys, os

HOST = "127.0.0.1"
PORT = 9876


class MCPError(Exception):
    """Blender MCP connection or protocol error."""
    pass


def send(code: str, strict_json: bool = True) -> dict:
    req = json.dumps({"type": "execute", "code": code, "strict_json": strict_json})
    try:
        s = socket.socket()
        s.settimeout(15)
        s.connect((HOST, PORT))
    except (ConnectionRefusedError, OSError) as exc:
        raise MCPError(
            f"Blender MCP server not reachable at {HOST}:{PORT} — "
            f"is Blender running with MCP enabled? ({exc})"
        ) from exc
    s.sendall((req + "\0").encode())
    data = b""
    while True:
        try:
            chunk = s.recv(4096)
        except socket.timeout:
            raise MCPError("MCP server timed out (15s). Split long operations into smaller scripts.")
        if not chunk:
            break
        data += chunk
        if b"\0" in data:
            break
    s.close()
    if not data:
        raise MCPError("MCP server returned empty response")
    try:
        return json.loads(data.decode().rstrip("\0"))
    except json.JSONDecodeError as exc:
        raise MCPError(f"MCP response is not valid JSON: {exc}") from exc


def run(code: str) -> None:
    """Send code to MCP, print result JSON to stdout, exit 1 on any error.

    One-line replacement for the common ``if __name__ == "__main__"`` pattern::

        from mcp_fetch import run
        run(code)
    """
    try:
        resp = send(code)
    except MCPError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(resp, indent=2))
    if resp.get("status") == "error":
        print(resp.get("message", ""), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # default: quick scene overview
        code = """
import bpy
ts_loaded = "bl_ext.user_default.texturesynth" in bpy.context.preferences.addons
ng_count = sum(1 for ng in bpy.data.node_groups
               if getattr(ng, "bl_idname", None) == "TextureSynthTreeType")
result = {
    "scene": bpy.context.scene.name,
    "ver": bpy.app.version_string,
    "ts_addon_loaded": ts_loaded,
    "ts_node_trees": ng_count,
}
"""
    else:
        arg = sys.argv[1]
        if os.path.isfile(arg):
            code = open(arg, "r").read()
        else:
            code = arg
    try:
        resp = send(code)
    except MCPError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        sys.exit(1)
    print(json.dumps(resp, indent=2))
    if resp.get("status") == "error":
        print(resp.get("message", ""), file=sys.stderr)
        sys.exit(1)
