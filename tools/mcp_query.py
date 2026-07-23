import socket, json

code = """
import bpy
result = {}
from bl_ext.user_default.texturesynth.core import engine_bridge as eb
from bl_ext.user_default.texturesynth.core import cpp_module as cpp
result['gen'] = str(eb.submitted_generation)
eng = cpp.get_engine()
result['eng'] = str(type(eng))
tree = None
for s in bpy.context.window.screen.areas:
    if s.type == 'NODE_EDITOR':
        sp = s.spaces.active
        if sp.node_tree and 'TextureSynth' in sp.node_tree.name:
            tree = sp.node_tree
            break
if tree and eng:
    a = tree.nodes.active
    result['act'] = a.name if a else 'None'
    u = getattr(a, 'ts_uuid', '') if a else ''
    result['uuid'] = u
    if u:
        n = int(u.replace('-', '')[:16], 16)
        result['nid'] = n
        g = eng.set_active_node(n)
        result['set_gen'] = str(g)
"""

req = json.dumps({"type": "execute", "code": code, "strict_json": True})
s = socket.socket(); s.settimeout(30); s.connect(("127.0.0.1", 9876))
s.sendall((req + "\0").encode())
data = b""
while True:
    try:
        chunk = s.recv(4096)
        if not chunk: break
        data += chunk
        if b"\0" in data: break
    except socket.timeout:
        break
s.close()
print(json.loads(data.decode().rstrip("\0")))
