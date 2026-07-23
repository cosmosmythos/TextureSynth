"""Auto-generates Blender node classes from the C++ NodeLibrary or local JSON fallback manifests."""
import json
import os

import bpy
from .base import TextureSynthNode, CHANNEL_MODE_ITEMS
from .tree import socket_type_for_format
from . import specialized
from ..utils.rna import register_class, unregister_class

_cf = None


def _format_to_string(format_value):
    """Normalize a socket format to a lowercase string.

    Handles both C++ ChannelFormat enums (ChannelFormat.Mono) and
    JSON fallback strings ('float', 'vec4', etc.).
    """
    if hasattr(format_value, 'name'):
        return format_value.name.lower()
    return str(format_value).lower()


def _socket_default(socket_info):
    """Get the default value from a socket, handling both C++ (default_value) and JSON (default)."""
    value = getattr(socket_info, 'default_value', None)
    if value is not None:
        return value
    return getattr(socket_info, 'default', None)


def _is_float_input(socket_info):
    """True when a socket carries a scalar float (manifest float input).

    Handles C++ Socket objects (SocketType.Float + ChannelFormat.Mono) and
    JSON fallback _JSONSocket objects (format='float' string).
    """
    socket_type = getattr(socket_info, 'type', None)
    if hasattr(socket_type, 'name') and socket_type.name == 'Float':
        return True
    return _format_to_string(getattr(socket_info, 'format', '')) == 'float'


_FORMAT_NAME_TO_OVERRIDE = {
    'mono':    'MONO',
    'uv':      'UV',
    'rgb':     'RGB',
    'rgba':    'RGBA',
    'vec4':    'DEFAULT',
    'float':   'MONO',
    'default': 'DEFAULT',
}


def _channel_format_to_override(channel_format):
    """Map a ChannelFormat enum value to a format_override string."""
    global _cf
    if _cf is None:
        from ..core import cpp_module
        _cf = cpp_module.get_core().ChannelFormat
    if channel_format == _cf.Mono:
        return 'MONO'
    if channel_format == _cf.UV:
        return 'UV'
    if channel_format == _cf.RGB:
        return 'RGB'
    return 'DEFAULT'


class _JSONParam:
    __slots__ = ('name', 'display_name', 'description',
                 'default_value', 'min_value', 'max_value',
                 'soft_min_value', 'soft_max_value',
                 'step', 'is_integer', 'as_socket', 'enum_values')
    def __init__(self, param_data):
        self.name        = param_data['name']
        self.display_name = param_data.get('display_name', self.name.replace('_', ' ').title())
        self.description = param_data.get('description', '')
        self.default_value = float(param_data.get('default', 0.0))
        self.min_value   = float(param_data.get('min', 0.0))
        self.max_value   = float(param_data.get('max', 1.0))
        self.soft_min_value = float(param_data.get('soft_min', self.min_value))
        self.soft_max_value = float(param_data.get('soft_max', self.max_value))
        self.step        = float(param_data.get('step', 0.0))
        self.is_integer  = bool(param_data.get('integer', False))
        self.as_socket   = bool(param_data.get('as_socket', False))
        self.enum_values = param_data.get('enum', [])


class _JSONSocket:
    __slots__ = ('name', 'format', 'default')
    def __init__(self, socket_data):
        self.name   = socket_data['name']
        self.format = socket_data.get('type', 'vec4')
        self.default = socket_data.get('default', None)


class _JSONNodeType:
    __slots__ = ('id', 'display_name', 'params', 'inputs', 'outputs')
    def __init__(self, node_data):
        self.id           = node_data['id']
        self.display_name = node_data.get('display_name', self.id.capitalize())
        self.params       = [_JSONParam(param_dict) for param_dict in node_data.get('params', [])]
        self.inputs       = [_JSONSocket(socket_data) for socket_data in node_data.get('inputs', [])]
        self.outputs      = [_JSONSocket(socket_data) for socket_data in node_data.get('outputs', [])]


def _load_manifest_fallback():
    """Read shader_assets/nodes/*.node.json from disk when C++ engine is unavailable."""
    addon_root = os.path.dirname(os.path.dirname(__file__))
    nodes_dir = os.path.join(addon_root, 'shader_assets', 'nodes')
    if not os.path.isdir(nodes_dir):
        return {}
    node_types = {}
    for file_name in sorted(os.listdir(nodes_dir)):
        if not file_name.endswith('.node.json'):
            continue
        file_path = os.path.join(nodes_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as file_handle:
                manifest_data = json.load(file_handle)
            node_types[manifest_data['id']] = _JSONNodeType(manifest_data)
        except Exception:
            pass
    return node_types


def _format_name_to_override(name):
    return _FORMAT_NAME_TO_OVERRIDE.get(name.lower(), 'DEFAULT')


_CATEGORY_BY_SVTYPE = {
    'perlin':'NOISE','simplex':'NOISE','worley':'NOISE','gabor':'NOISE',
    'value':'NOISE','white':'NOISE',
    'color_const':'INPUT','image':'INPUT',
    'blend':'BLEND',
    'invert':'FILTER','grayscale':'FILTER',
    'combine_rgba':'COLOR','separate_rgba':'COLOR',
    'shuffle':'CHANNEL','remap':'CHANNEL',
}

_FORMAT_OVERRIDE_SV_TYPES = {
    'color_const',
    'perlin', 'simplex', 'worley', 'gabor', 'value', 'white',
}

# Enum-backed node params. The integer index in the items tuple is the value
# stored in the SSBO; _ENUM_INDEX maps the string key back to that index.
_CHANNEL_ITEMS = [
    ('R', "R", ""),
    ('G', "G", ""),
    ('B', "B", ""),
    ('A', "A", ""),
]
_ENUM_PARAMS = {
    'channel_mode': CHANNEL_MODE_ITEMS,
    'r_src': _CHANNEL_ITEMS,
    'g_src': _CHANNEL_ITEMS,
    'b_src': _CHANNEL_ITEMS,
    'a_src': _CHANNEL_ITEMS,
}
_ENUM_INDEX = {k: {item[0]: i for i, item in enumerate(v)} for k, v in _ENUM_PARAMS.items()}

_generated_classes = []


def _supports_format_override(type_id, node_type):
    """Only expose format override on noise/input nodes with a single output socket."""
    return (
        type_id in _FORMAT_OVERRIDE_SV_TYPES
        and len(getattr(node_type, "outputs", [])) == 1
    )


def _update_param(self, context):
    try:
        from ..core.evaluation import request_param_update
        request_param_update()
    except Exception:
        pass


def generate_node_classes(core_module):
    """Dynamically generate bpy.types.Node classes from NodeLibrary or JSON fallback manifests."""
    global _generated_classes
    _generated_classes.clear()

    engine = core_module.get_engine()
    use_json_fallback = engine is None
    if use_json_fallback:
        all_types = _load_manifest_fallback()
    else:
        lib = engine.node_library()
        all_types = lib.all()

    if not all_types:
        _generated_classes.extend(specialized.collect_node_classes())
        return

    skip = specialized.specialized_sv_types()

    for type_id, node_type in all_types.items():
        if type_id in skip:
            continue

        class_name = f"TS_{type_id.capitalize()}_Node"
        as_socket_set = frozenset(
            param.name for param in node_type.params if getattr(param, 'as_socket', False)
        )
        class_dict = {
            'bl_idname': class_name,
            'bl_label':  node_type.display_name,
            'sv_type':   type_id,
            'ts_category': _CATEGORY_BY_SVTYPE.get(type_id, 'FILTER'),
            'supports_format_override': _supports_format_override(type_id, node_type),
            '_as_socket_names': as_socket_set,
            '__annotations__': {},
        }

        param_names = []
        dynamic_enum_index = {}

        for param in node_type.params:
            param_names.append(param.name)
            is_int = bool(getattr(param, "is_integer", False))
            step   = float(getattr(param, "step", 0.0))
            label  = getattr(param, "display_name", None)
            if label is None:
                label = param.name.replace('_', ' ').title()
            desc   = getattr(param, "description", "")

            enum_values = getattr(param, 'enum_values', None) or []
            if enum_values:
                items = [(str(i), name, "") for i, name in enumerate(enum_values)]
                default_key = str(int(round(param.default_value)))
                dynamic_enum_index[param.name] = {str(i): i for i in range(len(enum_values))}
                prop = bpy.props.EnumProperty(
                    name=label, description=desc,
                    items=items,
                    default=default_key,
                    update=_update_param,
                )
            elif param.name in _ENUM_PARAMS:
                items = _ENUM_PARAMS[param.name]
                default_key = items[int(round(param.default_value))][0]
                prop = bpy.props.EnumProperty(
                    name=label, description=desc,
                    items=items,
                    default=default_key,
                    update=_update_param,
                )
            elif is_int:
                prop = bpy.props.IntProperty(
                    name=label, description=desc,
                    default=int(round(param.default_value)),
                    min=int(round(param.min_value)),
                    max=int(round(param.max_value)),
                    soft_min=int(round(param.soft_min_value)),
                    soft_max=int(round(param.soft_max_value)),
                    update=_update_param,
                )
            else:
                is_as_socket = getattr(param, 'as_socket', False)
                prop = bpy.props.FloatProperty(
                    name=label, description=desc,
                    default=param.default_value,
                    min=param.min_value, max=param.max_value,
                    soft_min=param.soft_min_value,
                    soft_max=param.soft_max_value,
                    step=max(int(step * 100), 1) if step > 0 else 10,
                    precision=3,
                    subtype='FACTOR' if is_as_socket else 'NONE',
                    update=_update_param,
                )
            class_dict['__annotations__'][param.name] = prop

        # FloatProperty per float input so the socket draw() renders an inline slider when unlinked.
        for inp in node_type.inputs:
            if not _is_float_input(inp) or _socket_default(inp) is None:
                continue
            class_dict['__annotations__'][inp.name] = bpy.props.FloatProperty(
                name=inp.name.replace('_', ' ').title(),
                default=float(_socket_default(inp)),
                min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0,
                precision=3,
                subtype='FACTOR',
                update=_update_param,
            )

        # init: create sockets + UUID.
        def make_init(node_type_ref):
            def init_func(self, context):
                self.ts_category = _CATEGORY_BY_SVTYPE.get(self.sv_type, 'FILTER')
                super(type(self), self).init(context)
                single_in = len(node_type_ref.inputs) == 1 and not any(
                    getattr(param, 'as_socket', False) for param in node_type_ref.params)
                allow_format_override = getattr(type(self), 'supports_format_override', False)

                as_set = set()
                for param in node_type_ref.params:
                    if getattr(param, 'as_socket', False):
                        new_socket = self.inputs.new('TS_DefaultSocketType',
                                                     getattr(param, 'display_name', None) or param.name)
                        new_socket.name = param.name
                        as_set.add(param.name)
                self._as_socket_names = frozenset(as_set)
                for sock in node_type_ref.inputs:
                    display_label = "" if single_in else sock.name.replace('_', ' ').title()
                    fmt = getattr(sock, 'format', 'vec4')
                    if hasattr(fmt, 'name'):
                        override = _channel_format_to_override(fmt)
                    else:
                        override = _format_name_to_override(fmt)
                    in_type = socket_type_for_format(override)
                    new_sock = self.inputs.new(in_type, display_label)
                    if _is_float_input(sock) and _socket_default(sock) is not None:
                        setattr(self, sock.name, float(_socket_default(sock)))

                if node_type_ref.outputs:
                    fmt = getattr(node_type_ref.outputs[0], 'format', 'vec4')
                    if hasattr(fmt, 'name'):
                        fmt_override = _channel_format_to_override(fmt)
                    else:
                        fmt_override = _format_name_to_override(fmt)
                    if allow_format_override:
                        self.format_override = fmt_override
                meta = {}
                single_out = len(node_type_ref.outputs) == 1
                for i, sock in enumerate(node_type_ref.outputs):
                    label = "" if single_out else sock.name.replace('_', ' ').title()
                    if allow_format_override:
                        out_type = socket_type_for_format(
                            getattr(self, 'format_override', 'DEFAULT'))
                    else:
                        sock_fmt = getattr(sock, 'format', 'vec4')
                        if hasattr(sock_fmt, 'name'):
                            ov = _channel_format_to_override(sock_fmt)
                        else:
                            ov = _format_name_to_override(sock_fmt)
                        out_type = socket_type_for_format(ov)
                    self.outputs.new(out_type, label)
                    meta[i] = (sock.name, out_type)
                self._ts_output_meta = meta
            return init_func
        class_dict['init'] = make_init(node_type)

        # draw_buttons.
        def make_draw_buttons(p_names, p_as_socket):
            def draw_buttons_func(self, context, layout):
                self.draw_error_ui(layout)
                self.draw_format_override_ui(layout)
                for p_name, is_sock in zip(p_names, p_as_socket):
                    if is_sock:
                        continue  # socket draw() handles inline slider
                    layout.prop(self, p_name)
            return draw_buttons_func
        p_as_socket = [getattr(param, 'as_socket', False) for param in node_type.params]
        class_dict['draw_buttons'] = make_draw_buttons(param_names, p_as_socket)

        # Float input names+defaults for SSBO layout alignment with the fused emitter.
        float_input_specs = [(inp.name, float(_socket_default(inp)))
                             for inp in node_type.inputs
                             if _is_float_input(inp) and _socket_default(inp) is not None]

        # get_parameters positional mapping. SSBO layout: [manifest_params..., float_input_defaults...].
        # Enum params are stored as their integer index via _ENUM_INDEX.
        def make_get_parameters(p_names, float_specs, enum_indices):
            def get_parameters_func(self):
                result = []
                for param_name in p_names:
                    value = getattr(self, param_name)
                    if param_name in enum_indices:
                        result.append(float(enum_indices[param_name].get(value, 0)))
                    else:
                        result.append(float(value))
                for name, default in float_specs:
                    result.append(float(getattr(self, name, default)))
                return result
            return get_parameters_func
        merged_enum_index = {**_ENUM_INDEX, **dynamic_enum_index}
        class_dict['get_parameters'] = make_get_parameters(param_names, float_input_specs, merged_enum_index)

        # get_named_parameters returns None when float inputs exist, so _push_params_using_stable_ids
        # falls through to get_parameters() which writes both manifest params AND float-input slots.
        def make_get_named_parameters(p_names, has_float_inputs):
            def get_named_parameters_func(self):
                if has_float_inputs:
                    return None
                return {param_name: float(getattr(self, param_name)) for param_name in p_names}
            return get_named_parameters_func
        class_dict['get_named_parameters'] = make_get_named_parameters(param_names, bool(float_input_specs))

        _generated_classes.append(type(class_name, (TextureSynthNode,), class_dict))

    _generated_classes.extend(specialized.collect_node_classes())


def get_generated_classes():
    return _generated_classes


def register():
    # Register PropertyGroups and sockets before node classes that reference them.
    for pg in specialized.specialized_property_groups():
        register_class(pg)

    for sock in specialized.collect_socket_classes():
        register_class(sock)

    for cls in _generated_classes:
        register_class(cls)

    for op in specialized.collect_operator_classes():
        register_class(op)


def unregister():
    for op in reversed(specialized.collect_operator_classes()):
        unregister_class(op)
    for cls in reversed(_generated_classes):
        unregister_class(cls)
    for sock in reversed(specialized.collect_socket_classes()):
        unregister_class(sock)
    for pg in reversed(specialized.specialized_property_groups()):
        unregister_class(pg)
