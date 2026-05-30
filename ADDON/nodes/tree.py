"""
Node Tree and Socket definitions for TextureSynth.

Pure UI — no engine logic, no imports from core/.
"""

import bpy


# ── Format-specific socket types ──────────────────────────────────
# Each format gets its own socket class with a distinct color.
# The draw() behavior is identical: label when linked, inline prop
# when unlinked (so the user can set a default value without a link).

def _make_socket_draw(bl_idname):
    """Return a draw() method that shows an inline node prop when unlinked."""
    def draw(self, context, layout, node, text):
        if self.is_linked:
            layout.label(text=text)
        elif node and hasattr(node, self.name):
            layout.prop(node, self.name, text=text)
        else:
            layout.label(text=text)
    return draw


class TS_DefaultSocket(bpy.types.NodeSocket):
    """Default texture socket — used when no format override is set."""
    bl_idname = 'TS_DefaultSocketType'
    bl_label = "Texture Socket"
    draw = _make_socket_draw('TS_DefaultSocketType')

    @classmethod
    def draw_color_simple(cls):
        return (0.78, 0.78, 0.2, 1.0)


class TS_MonoSocket(bpy.types.NodeSocket):
    """Mono / Float output — single-channel grayscale."""
    bl_idname = 'TS_MonoSocketType'
    bl_label = "Mono Socket"
    draw = _make_socket_draw('TS_MonoSocketType')

    @classmethod
    def draw_color_simple(cls):
        return (0.5, 0.5, 0.5, 1.0)


class TS_UVSocket(bpy.types.NodeSocket):
    """UV / Vector output — two-channel coordinate data."""
    bl_idname = 'TS_UVSocketType'
    bl_label = "UV Socket"
    draw = _make_socket_draw('TS_UVSocketType')

    @classmethod
    def draw_color_simple(cls):
        return (0.39, 0.39, 0.78, 1.0)


class TS_ColorSocket(bpy.types.NodeSocket):
    """Color / RGBA output — full four-channel color data."""
    bl_idname = 'TS_ColorSocketType'
    bl_label = "Color Socket"
    draw = _make_socket_draw('TS_ColorSocketType')

    @classmethod
    def draw_color_simple(cls):
        return (0.78, 0.78, 0.16, 1.0)


class TS_IntSocket(bpy.types.NodeSocket):
    """Integer ID output — integer标识 data."""
    bl_idname = 'TS_IntSocketType'
    bl_label = "ID Socket"
    draw = _make_socket_draw('TS_IntSocketType')

    @classmethod
    def draw_color_simple(cls):
        return (0.12, 0.6, 0.12, 1.0)


# Backwards-compatible alias
TextureSynthSocket = TS_DefaultSocket


# ── Format ↔ Socket type mapping ──────────────────────────────────

FORMAT_SOCKET_MAP = {
    'DEFAULT': 'TS_DefaultSocketType',
    'MONO':    'TS_MonoSocketType',
    'UV':      'TS_UVSocketType',
    'RGB':     'TS_ColorSocketType',
    'RGBA':    'TS_ColorSocketType',
    'ID':      'TS_IntSocketType',
}

def _get_channel_format():
    """Lazy import ChannelFormat enum from C++ module."""
    from ..core import cpp_module
    core = cpp_module.get_core()
    return core.ChannelFormat

FORMAT_CHANNEL_MAP = {
    'DEFAULT': None,
    'MONO':    lambda: _get_channel_format().Mono,
    'UV':      lambda: _get_channel_format().UV,
    'RGB':     lambda: _get_channel_format().RGB,
    'RGBA':    lambda: _get_channel_format().RGBA,
    'ID':      lambda: _get_channel_format().ID,
}


def socket_type_for_format(fmt):
    """Return the bl_idname for a given format string."""
    return FORMAT_SOCKET_MAP.get(fmt, 'TS_DefaultSocketType')


def replace_socket(socket, new_type, new_name=None):
    """Replace a socket with a new type, preserving links and position.

    Returns the new socket. The old socket reference becomes invalid.
    """
    socket_name = new_name or socket.name
    sockets = socket.node.outputs if socket.is_output else socket.node.inputs
    socket_pos = list(sockets).index(socket)

    if socket.is_output:
        to_sockets = [l.to_socket for l in socket.links]
        socket.node.outputs.remove(socket)
        new_socket = socket.node.outputs.new(new_type, socket_name)
        socket.node.outputs.move(len(socket.node.outputs) - 1, socket_pos)
        for to_socket in to_sockets:
            socket.id_data.links.new(new_socket, to_socket)
    else:
        from_socket = socket.links[0].from_socket if socket.is_linked else None
        socket.node.inputs.remove(socket)
        new_socket = socket.node.inputs.new(new_type, socket_name)
        socket.node.inputs.move(len(socket.node.inputs) - 1, socket_pos)
        if from_socket:
            socket.id_data.links.new(from_socket, new_socket)

    return new_socket


# ── Node Tree ─────────────────────────────────────────────────────

class TextureSynthTree(bpy.types.NodeTree):
    """Custom node tree for procedural texture authoring."""
    bl_idname = 'TextureSynthTreeType'
    bl_label = "TextureSynth Editor"
    bl_icon = 'TEXTURE'

    @classmethod
    def poll(cls, context):
        return True

    def update(self):
        """Called when connections are made/broken (topology change)."""
        try:
            from ..core.evaluation import request_topology_update
            request_topology_update()
        except Exception:
            pass


# ── Registration ──────────────────────────────────────────────────

classes = (
    TS_DefaultSocket,
    TS_MonoSocket,
    TS_UVSocket,
    TS_ColorSocket,
    TS_IntSocket,
    TextureSynthTree,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
