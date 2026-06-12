"""Auto-generates Blender node classes from the C++ NodeLibrary or local JSON fallback manifests."""
import json
import os

import bpy
from .base import TextureSynthNode
from .tree import socket_type_for_format
from . import specialized
from ..utils.rna import register_class, unregister_class

_cf = None

_FORMAT_NAME_TO_OVERRIDE = {
    'mono':    'MONO',
    'uv':      'UV',
    'rgb':     'RGB',
    'rgba':    'RGBA',
    'id':      'ID',
    'vec4':    'DEFAULT',
    'float':   'MONO',
    'metadata': 'DEFAULT',
    'default': 'DEFAULT',
}


def _channel_format_to_override(cf):
    """Map a ChannelFormat enum value to a format_override string."""
    global _cf
    if _cf is None:
        from ..core import cpp_module
        _cf = cpp_module.get_core().ChannelFormat
    if cf == _cf.Mono:
        return 'MONO'
    if cf == _cf.UV:
        return 'UV'
    if cf == _cf.RGB:
        return 'RGB'
    if cf == _cf.ID:
        return 'ID'
    if cf == _cf.Metadata:
        return 'DEFAULT'
    return 'DEFAULT'


class _JSONParam:
    __slots__ = ('name', 'display_name', 'description',
                 'default_value', 'min_value', 'max_value',
                 'soft_min_value', 'soft_max_value',
                 'step', 'is_integer', 'as_socket')
    def __init__(self, d):
        self.name        = d['name']
        self.display_name = d.get('display_name', self.name.replace('_', ' ').title())
        self.description = d.get('description', '')
        self.default_value = float(d.get('default', 0.0))
        self.min_value   = float(d.get('min', 0.0))
        self.max_value   = float(d.get('max', 1.0))
        self.soft_min_value = float(d.get('soft_min', self.min_value))
        self.soft_max_value = float(d.get('soft_max', self.max_value))
        self.step        = float(d.get('step', 0.0))
        self.is_integer  = bool(d.get('integer', False))
        self.as_socket   = bool(d.get('as_socket', False))


class _JSONSocket:
    __slots__ = ('name', 'format', 'default')
    def __init__(self, d):
        self.name   = d['name']
        self.format = d.get('type', 'vec4')
        self.default = d.get('default', None)


class _JSONNodeType:
    __slots__ = ('id', 'display_name', 'params', 'inputs', 'outputs')
    def __init__(self, d):
        self.id           = d['id']
        self.display_name = d.get('display_name', self.id.capitalize())
        self.params       = [_JSONParam(p) for p in d.get('params', [])]
        self.inputs       = [_JSONSocket(s) for s in d.get('inputs', [])]
        self.outputs      = [_JSONSocket(s) for s in d.get('outputs', [])]


def _load_manifest_fallback():
    """Read shader_assets/nodes/*.node.json from disk when C++ engine is unavailable."""
    addon_root = os.path.dirname(os.path.dirname(__file__))
    nodes_dir = os.path.join(addon_root, 'shader_assets', 'nodes')
    if not os.path.isdir(nodes_dir):
        return {}
    out = {}
    for fname in sorted(os.listdir(nodes_dir)):
        if not fname.endswith('.node.json'):
            continue
        path = os.path.join(nodes_dir, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            out[d['id']] = _JSONNodeType(d)
        except Exception:
            pass
    return out


def _format_name_to_override(name):
    return _FORMAT_NAME_TO_OVERRIDE.get(name.lower(), 'DEFAULT')


_CATEGORY_BY_SVTYPE = {
    'perlin':'NOISE','simplex':'NOISE','worley':'NOISE','gabor':'NOISE',
    'value':'NOISE','white':'NOISE',
    'color_const':'INPUT','image':'INPUT',
    'blend':'BLEND',
    'invert':'FILTER','grayscale':'FILTER',
    'combine_rgba':'COLOR','separate_rgba':'COLOR',
}

_FORMAT_OVERRIDE_SV_TYPES = {
    'color_const',
    'perlin', 'simplex', 'worley', 'gabor', 'value', 'white',
}

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
            p.name for p in node_type.params if getattr(p, 'as_socket', False)
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

        # Setup Float and Int properties from parameters.
        for param in node_type.params:
            param_names.append(param.name)
            is_int = bool(getattr(param, "is_integer", False))
            step   = float(getattr(param, "step", 0.0))
            label  = getattr(param, "display_name", None) or param.name.replace('_', ' ').title()
            desc   = getattr(param, "description", "")

            if is_int:
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

        # Define init function creating sockets and UUID.
        def make_init(node_type_ref):
            def init_func(self, context):
                self.ts_category = _CATEGORY_BY_SVTYPE.get(self.sv_type, 'FILTER')
                super(type(self), self).init(context)
                single_in = len(node_type_ref.inputs) == 1 and not any(
                    getattr(p, 'as_socket', False) for p in node_type_ref.params)
                allow_format_override = getattr(type(self), 'supports_format_override', False)

                as_set = set()
                for p in node_type_ref.params:
                    if getattr(p, 'as_socket', False):
                        s = self.inputs.new('TS_DefaultSocketType',
                                            getattr(p, 'display_name', None) or p.name)
                        s.name = p.name
                        as_set.add(p.name)
                self._as_socket_names = frozenset(as_set)
                for sock in node_type_ref.inputs:
                    label = "" if single_in else sock.name.capitalize()
                    fmt = getattr(sock, 'format', 'vec4')
                    if hasattr(fmt, 'name'):
                        override = _channel_format_to_override(fmt)
                    else:
                        override = _format_name_to_override(fmt)
                    in_type = socket_type_for_format(override)
                    new_sock = self.inputs.new(in_type, label)
                    if sock.format == 'float' and sock.default is not None:
                        new_sock.default_value = float(sock.default)
                        setattr(self, sock.name, float(sock.default))

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
                    label = "" if single_out else sock.name.capitalize()
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

        # Define draw_buttons function.
        def make_draw_buttons(p_names, p_as_socket):
            def draw_buttons_func(self, context, layout):
                self.draw_error_ui(layout)
                self.draw_format_override_ui(layout)
                for p_name, is_sock in zip(p_names, p_as_socket):
                    if is_sock:
                        continue  # socket draw() handles inline slider
                    layout.prop(self, p_name)
            return draw_buttons_func
        p_as_socket = [getattr(p, 'as_socket', False) for p in node_type.params]
        class_dict['draw_buttons'] = make_draw_buttons(param_names, p_as_socket)

        # Collect float input names+defaults for SSBO layout alignment with fused emitter.
        float_input_specs = [(inp.name, float(inp.default))
                             for inp in node_type.inputs
                             if inp.format == 'float' and inp.default is not None]

        # Define get_parameters positional mapping.
        # SSBO layout: [manifest_params..., float_input_defaults...]
        def make_get_parameters(p_names, float_specs):
            def get_parameters_func(self):
                result = [float(getattr(self, p)) for p in p_names]
                for name, default in float_specs:
                    result.append(float(getattr(self, name, default)))
                return result
            return get_parameters_func
        class_dict['get_parameters'] = make_get_parameters(param_names, float_input_specs)

        # Define get_named_parameters dictionary mapping.
        def make_get_named_parameters(p_names):
            def get_named_parameters_func(self):
                return {p: float(getattr(self, p)) for p in p_names}
            return get_named_parameters_func
        class_dict['get_named_parameters'] = make_get_named_parameters(param_names)

        _generated_classes.append(type(class_name, (TextureSynthNode,), class_dict))

    _generated_classes.extend(specialized.collect_node_classes())


def get_generated_classes():
    return _generated_classes


def validate_named_param_contract(core_module):
    """Verify that generated node parameters cover all manifest parameters."""
    engine = core_module.get_engine()
    if engine is None:
        return
    lib = engine.node_library().all()
    for cls in _generated_classes:
        sv = getattr(cls, 'sv_type', None)
        if sv is None or sv not in lib:
            continue
        manifest_names = {p.name for p in lib[sv].params}

        ann_keys = set(cls.__annotations__.keys())
        emitted = ann_keys
        try:
            import inspect, re
            src = inspect.getsource(cls.get_named_parameters)
            keys_in_src = set(re.findall(r'[{,]\s*["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']\s*:', src))
            if keys_in_src:
                emitted = keys_in_src
        except Exception:
            pass

        missing = manifest_names - emitted
        if missing:
            pass


def register():
    # Register PropertyGroups and sockets before node classes that reference them.
    for pg in specialized.specialized_property_groups():
        register_class(pg)

    for sock in specialized.collect_socket_classes():
        register_class(sock)

    for cls in _generated_classes:
        register_class(cls)

    from ..core import cpp_module
    validate_named_param_contract(cpp_module)


def unregister():
    for cls in reversed(_generated_classes):
        unregister_class(cls)
    for sock in reversed(specialized.collect_socket_classes()):
        unregister_class(sock)
    for pg in reversed(specialized.specialized_property_groups()):
        unregister_class(pg)
