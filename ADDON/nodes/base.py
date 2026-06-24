import bpy
import uuid


FORMAT_OVERRIDE_ITEMS = [
    ('DEFAULT', "Auto",    ""),
    ('MONO',    "Mono",    ""),
    ('UV',      "UV",      ""),
    ('RGB',     "RGB",     ""),
    ('RGBA',    "RGBA",    ""),
    ('ID',      "Integer", ""),
]

CHANNEL_MODE_ITEMS = [
    ('MONO', "Mono", ""),
    ('UV',   "UV",   ""),
    ('RGB',  "RGB",  ""),
    ('RGBA', "RGBA", ""),
]

DEPTH_MODE_ITEMS = [
    ('AUTO',       "Auto",        "Use graph default"),
    ('MATCH_INPUT', "Match Input", "Match input bit depth"),
    ('ABSOLUTE',   "Absolute",    "Use specific bit depth"),
]

ABSOLUTE_DEPTH_ITEMS = [
    ('F8',  "8-bit",  ""),
    ('F16', "16f",    ""),
    ('F32', "32f",    ""),
]


def _resolve_depth_enum(depth_str):
    """Map the absolute_depth string ('F8'/'F16'/'F32') to the cpp BitDepth enum."""
    try:
        from ..core import cpp_module
        BitDepth = cpp_module.BitDepth
        return {'F8': BitDepth.F8, 'F16': BitDepth.F16, 'F32': BitDepth.F32}.get(depth_str, BitDepth.F16)
    except Exception:
        return None


def _resolve_depth_mode_enum(mode_str):
    try:
        from ..core import cpp_module
        DepthMode = cpp_module.DepthMode
        return {
            'AUTO':        DepthMode.Auto,
            'MATCH_INPUT': DepthMode.MatchInput,
            'ABSOLUTE':    DepthMode.Absolute,
        }.get(mode_str, DepthMode.Auto)
    except Exception:
        return None


def _on_format_override_change(self, context):
    try:
        from ..core.evaluation import request_topology_update
        request_topology_update()
    except Exception:
        pass
    if not getattr(self, '_deferred_fmt_rebuild', False):
        self._deferred_fmt_rebuild = True
        def _rebuild(self=self):
            self.rebuild_output_sockets()
            setattr(self, '_deferred_fmt_rebuild', False)
        bpy.app.timers.register(_rebuild, first_interval=0.0)


class TextureSynthNode(bpy.types.Node):
    """Base class for all TextureSynth nodes."""

    ts_uuid: bpy.props.StringProperty(
        name="TextureSynth UUID",
        description="Internal ID (auto-generated)",
        default="",
        options={'HIDDEN'},
    )

    ts_compile_error: bpy.props.StringProperty(
        name="Compile Error",
        description="Last compile error (read-only)",
        default="",
        options={'HIDDEN'},
    )

    format_override: bpy.props.EnumProperty(
        name="Format",
        description="Override output format",
        items=FORMAT_OVERRIDE_ITEMS,
        default='DEFAULT',
        update=_on_format_override_change,
    )

    depth_mode: bpy.props.EnumProperty(
        name="Depth",
        description="How bit depth is chosen",
        items=DEPTH_MODE_ITEMS,
        default='AUTO',
        update=_on_format_override_change,
    )

    absolute_depth: bpy.props.EnumProperty(
        name="Bit Depth",
        description="Bit depth when mode is Absolute",
        items=ABSOLUTE_DEPTH_ITEMS,
        default='F16',
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

    def get_content_signature(self):
        """Non-slider state affecting output (image pixels, frame ticks).
        Override on content-bearing nodes. None = slider-only."""
        return None

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

    def get_depth_mode(self):
        """Return DepthMode enum value for engine (Auto if not supported)."""
        if not getattr(self, 'supports_format_override', False):
            return None
        return _resolve_depth_mode_enum(getattr(self, 'depth_mode', 'AUTO'))

    def get_absolute_depth(self):
        """Return BitDepth enum value for engine (F16 default)."""
        return _resolve_depth_enum(getattr(self, 'absolute_depth', 'F16'))

    def draw_format_override_ui(self, layout):
        """Draw format override + depth controls for supported nodes."""
        if not getattr(self, 'supports_format_override', False):
            return
        layout.prop(self, 'format_override', text="")
        col = layout.column(align=True)
        col.prop(self, 'depth_mode', text="")
        if getattr(self, 'depth_mode', 'AUTO') == 'ABSOLUTE':
            col.prop(self, 'absolute_depth', text="")
