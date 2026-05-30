"""
Image Input node.

Backed by shader_assets/nodes/image.node.json + image.glsl.

JSON param order (MUST be matched by get_parameters):
    [u_offset, v_offset, u_scale, v_scale]
"""
import bpy
from ..base import TextureSynthNode
from ._common import update_param


SV_TYPE = "image"   # matches image.node.json `id`


def update_image_prop(self, context):
    """
    Callback when the user selects or changes the image datablock.
    Uploads the image to Vulkan immediately, then triggers a fast render redispatch.
    """
    try:
        from ...core.engine_bridge import upload_node_image
        upload_node_image(self)
    except Exception as e:
        print(f"[TextureSynth] update_image_prop upload exception: {e}")
    
    # Request standard parameter rendering redispatch
    update_param(self, context)


class TS_Image_Node(TextureSynthNode):
    bl_idname = 'TS_Image_Node'
    bl_label  = 'Image Input'
    sv_type   = SV_TYPE

    image: bpy.props.PointerProperty(
        type=bpy.types.Image,
        name="Image",
        update=update_image_prop,
    )

    u_offset: bpy.props.FloatProperty(
        name="U Offset",
        default=0.0,
        update=update_param,
    )

    v_offset: bpy.props.FloatProperty(
        name="V Offset",
        default=0.0,
        update=update_param,
    )

    u_scale: bpy.props.FloatProperty(
        name="U Scale",
        default=1.0,
        min=0.001,
        update=update_param,
    )

    v_scale: bpy.props.FloatProperty(
        name="V Scale",
        default=1.0,
        min=0.001,
        update=update_param,
    )

    def init(self, context):
        super().init(context)              # assigns ts_uuid
        self.outputs.new('TS_DefaultSocketType', "Color")

    def draw_buttons(self, context, layout):
        self.draw_error_ui(layout)
        layout.prop(self, 'format_override', text="")
        layout.template_ID(self, "image", open="image.open")
        
        col = layout.column(align=True)
        col.prop(self, "u_offset")
        col.prop(self, "v_offset")
        
        col = layout.column(align=True)
        col.prop(self, "u_scale")
        col.prop(self, "v_scale")

    def get_parameters(self):
        # Order: [u_offset, v_offset, u_scale, v_scale]
        return [
            float(self.u_offset),
            float(self.v_offset),
            float(self.u_scale),
            float(self.v_scale),
        ]

    def get_named_parameters(self):
        return {"u_offset": float(self.u_offset),
                "v_offset": float(self.v_offset),
                "u_scale":  float(self.u_scale),
                "v_scale":  float(self.v_scale)}


NODE_CLASS = TS_Image_Node
