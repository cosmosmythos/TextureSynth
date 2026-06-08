import bpy
import uuid


FORMAT_OVERRIDE_ITEMS = [
    ('DEFAULT', "Auto",    "Use the node type's declared output format"),
    ('MONO',    "Mono",    "Single-channel grayscale (R16_SFLOAT)"),
    ('UV',      "UV",      "Two-channel vector (R16G16_SFLOAT)"),
    ('RGB',     "RGB",     "Three-channel color (R16G16B16A16_SFLOAT)"),
    ('RGBA',    "RGBA",    "Full four-channel color (R16G16B16A16_SFLOAT)"),
    ('ID',      "Integer", "Integer identifier (R32_UINT)"),
]


def _on_format_override_change(self, context):
    """Rebuild output sockets when format_override changes.
    Deferred to avoid crashing Blender during UI draw (replace_socket
    modifies the socket layout while Blender may be iterating it)."""
    try:
        from ..core.evaluation import request_topology_update
        request_topology_update()
    except Exception:
        pass
    if not getattr(self, '_deferred_fmt_rebuild', False):
        self._deferred_fmt_rebuild = True
        bpy.app.timers.register(
            lambda self=self: (self.rebuild_output_sockets(),
                               setattr(self, '_deferred_fmt_rebuild', False)) or None,
            first_interval=0.0)


class TextureSynthNode(bpy.types.Node):
    """Base class for all TextureSynth nodes."""

    ts_uuid: bpy.props.StringProperty(
        name="TextureSynth UUID",
        description="Stable internal node identity used by the TextureSynth engine",
        default="",
        options={'HIDDEN'},
    )

    ts_compile_error: bpy.props.StringProperty(
        name="Compile Error",
        description="Last shader compilation error for this node",
        default="",
        options={'HIDDEN'},
    )

    format_override: bpy.props.EnumProperty(
        name="Format",
        description="Override the output texture format for this node.",
        items=FORMAT_OVERRIDE_ITEMS,
        default='DEFAULT',
        update=_on_format_override_change,
    )

    def draw_error_ui(self, layout):
        if getattr(self, "ts_compile_error", ""):
            box = layout.box()
            box.alert = True
            row = box.row()
            row.label(text="Compile Failed!", icon='ERROR')
            row.prop(self, "ts_compile_error", text="")

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == 'TextureSynthTreeType'

    def ensure_ts_uuid(self):
        """Lazily assign a UUID if this node doesn't have one yet."""
        if not getattr(self, "ts_uuid", ""):
            self.ts_uuid = uuid.uuid4().hex
        return self.ts_uuid

    def stable_id(self) -> int:
        """Return a deterministic uint64 derived from this node's UUID."""
        h = self.ensure_ts_uuid()
        return int(h[:16], 16)

    ts_category: str = 'INPUT'
    supports_format_override: bool = False

    # Stored output socket metadata: maps socket_index -> (original_name, original_type)
    _ts_output_meta: dict = {}

    def init(self, context):
        """Called when the node is first created. Subclasses should call super()."""
        if not getattr(self, "ts_uuid", ""):
            self.ts_uuid = uuid.uuid4().hex

    def copy(self, node):
        """Called when the node is duplicated. Generate a fresh UUID."""
        self.ts_uuid = uuid.uuid4().hex

    def update(self):
        """Called when a property (slider) changes."""
        try:
            from ..core.evaluation import request_param_update
            request_param_update()
        except Exception:
            pass

    def rebuild_output_sockets(self):
        """Replace all output sockets to match the current format_override, preserving links."""
        if not getattr(self, 'supports_format_override', False):
            return
        from .tree import socket_type_for_format, replace_socket

        fmt = getattr(self, 'format_override', 'DEFAULT')
        new_type = socket_type_for_format(fmt)
        meta = getattr(self, '_ts_output_meta', {})

        for i, sock in enumerate(list(self.outputs)):
            orig_name, _ = meta.get(i, (sock.name, sock.bl_idname))
            if sock.bl_idname != new_type:
                replace_socket(sock, new_type, orig_name)
            meta[i] = (orig_name, new_type)
        self._ts_output_meta = meta

    def get_format_override(self):
        """Return ChannelFormat enum value for engine."""
        from .tree import FORMAT_CHANNEL_MAP, _get_channel_format
        if not getattr(self, 'supports_format_override', False):
            return _get_channel_format().RGBA
        fmt = getattr(self, 'format_override', 'DEFAULT')
        getter = FORMAT_CHANNEL_MAP.get(fmt)
        if getter:
            return getter()
        return _get_channel_format().RGBA

    def draw_format_override_ui(self, layout):
        """Draw format override only for nodes where it is meaningful."""
        if getattr(self, 'supports_format_override', False):
            layout.prop(self, 'format_override', text="")
