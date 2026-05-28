import bpy
import uuid

# Category color palette
TS_CATEGORY_COLORS = {
    'INPUT':   (0.20, 0.35, 0.55),   # blue-ish
    'NOISE':   (0.25, 0.45, 0.30),   # green-ish
    'COLOR':   (0.55, 0.40, 0.20),   # orange-ish
    'FILTER':  (0.45, 0.25, 0.45),   # purple-ish
    'BLEND':   (0.55, 0.30, 0.30),   # red-ish
    'OUTPUT':  (0.15, 0.15, 0.15),   # neutral dark
}

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
        """Return a deterministic uint64 derived from this node's UUID.

        The C++ engine uses NodeId = uint64_t. We take the first 16 hex
        characters (64 bits) of the 128-bit UUID — negligible collision
        probability for any realistic graph size.
        """
        h = self.ensure_ts_uuid()
        return int(h[:16], 16)

    ts_category: str = 'INPUT'

    def init(self, context):
        """Called when the node is first created. Subclasses should call super()."""
        if not getattr(self, "ts_uuid", ""):
            self.ts_uuid = uuid.uuid4().hex
        color = TS_CATEGORY_COLORS.get(self.ts_category)
        if color:
            self.use_custom_color = True
            self.color = color

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
