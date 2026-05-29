"""factory.py — auto-generates Blender node classes from C++ NodeLibrary.

Nodes that need a custom UI live under nodes/specialized/ and are skipped
here (by sv_type), then appended to _generated_classes after the loop.
"""
import bpy
from .base import TextureSynthNode
from . import specialized


_CATEGORY_BY_SVTYPE = {
    'perlin':'NOISE','simplex':'NOISE','worley':'NOISE','gabor':'NOISE',
    'value':'NOISE','white':'NOISE',
    'color_const':'INPUT','image':'INPUT',
    'blend':'BLEND',
    'invert':'FILTER','grayscale':'FILTER','rgba_merge':'FILTER',
}

_generated_classes = []


def _update_param(self, context):
    try:
        from ..core.evaluation import request_param_update
        request_param_update()
    except Exception:
        pass


def generate_node_classes(core_module):
    """Dynamically generate bpy.types.Node classes from the C++ NodeLibrary."""
    global _generated_classes
    _generated_classes.clear()

    engine = core_module.get_engine()
    if not engine:
        return

    lib = engine.node_library()
    all_types = lib.all()

    # sv_types owned by hand-written modules — never auto-generate these.
    skip = specialized.specialized_sv_types()

    for type_id, node_type in all_types.items():
        if type_id in skip:
            continue

        class_name = f"TS_{type_id.capitalize()}_Node"
        class_dict = {
            'bl_idname': class_name,
            'bl_label':  node_type.display_name,
            'sv_type':   type_id,
            '__annotations__': {},
        }

        param_names = []

        # 1. Properties (one per JSON param)
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
                    update=_update_param,
                )
            else:
                prop = bpy.props.FloatProperty(
                    name=label, description=desc,
                    default=param.default_value,
                    min=param.min_value, max=param.max_value,
                    step=max(int(step * 100), 1) if step > 0 else 10,
                    precision=3,
                    update=_update_param,
                )
            class_dict['__annotations__'][param.name] = prop

        # 2. init — sockets + UUID
        def make_init(node_type_ref):
            def init_func(self, context):
                self.ts_category = _CATEGORY_BY_SVTYPE.get(self.sv_type, 'FILTER')
                super(type(self), self).init(context)
                single_in = len(node_type_ref.inputs) == 1 and not any(
                    getattr(p, 'as_socket', False) for p in node_type_ref.params)
                for sock in node_type_ref.inputs:
                    label = "" if single_in else sock.name.capitalize()
                    self.inputs.new('TextureSynthSocketType', label)
                for p in node_type_ref.params:
                    if getattr(p, 'as_socket', False):
                        s = self.inputs.new('TextureSynthSocketType',
                                            getattr(p, 'display_name', None) or p.name)
                        s.name = p.name
                single_out = len(node_type_ref.outputs) == 1        
                for sock in node_type_ref.outputs:
                    label = "" if single_out else sock.name.capitalize()
                    self.outputs.new('TextureSynthSocketType', label)
            return init_func
        class_dict['init'] = make_init(node_type)

        # 3. draw_buttons — hide socket-driven params from node body
        def make_draw_buttons(p_names, p_as_socket):
            def draw_buttons_func(self, context, layout):
                self.draw_error_ui(layout)
                for p_name, is_sock in zip(p_names, p_as_socket):
                    if is_sock:
                        continue  # drawn inline by socket.draw() now
                    layout.prop(self, p_name)
            return draw_buttons_func
        p_as_socket = [getattr(p, 'as_socket', False) for p in node_type.params]
        class_dict['draw_buttons'] = make_draw_buttons(param_names, p_as_socket)

        # 4. get_parameters — list[float] in JSON order
        def make_get_parameters(p_names):
            def get_parameters_func(self):
                return [float(getattr(self, p)) for p in p_names]
            return get_parameters_func
        class_dict['get_parameters'] = make_get_parameters(param_names)

        # 4b. get_named_parameters — dict[name -> float] for Phase 6 name-based binding.
        def make_get_named_parameters(p_names):
            def get_named_parameters_func(self):
                return {p: float(getattr(self, p)) for p in p_names}
            return get_named_parameters_func
        class_dict['get_named_parameters'] = make_get_named_parameters(param_names)

        _generated_classes.append(type(class_name, (TextureSynthNode,), class_dict))

    # 5. Append every hand-written class (incl. Output marker)
    _generated_classes.extend(specialized.collect_node_classes())


def get_generated_classes():
    return _generated_classes


def validate_named_param_contract(core_module):
    """Startup check: every generated node's get_named_parameters() keys
    must be a subset of (or equal to) its C++ manifest param names.
    Specialized nodes are allowed to expose synthesized keys (e.g. r/g/b
    derived from a color picker) but must cover every manifest param."""
    engine = core_module.get_engine()
    if engine is None:
        return
    lib = engine.node_library().all()
    issues = 0
    checked = 0
    for cls in _generated_classes:
        sv = getattr(cls, 'sv_type', None)
        if sv is None or sv not in lib:
            continue
        checked += 1
        manifest_names = {p.name for p in lib[sv].params}

        # Auto-generated nodes: annotations ARE the keys.
        ann_keys = set(cls.__annotations__.keys())
        # Specialized nodes can over-cover (e.g. color_const has color_data
        # but emits r,g,b,a). Compute the *emitted* set if possible.
        emitted = ann_keys
        try:
            # Best-effort: parse the function source for literal keys.
            # Falls back to annotations if introspection fails.
            import inspect, re
            src = inspect.getsource(cls.get_named_parameters)
            keys_in_src = set(re.findall(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']\s*:', src))
            if keys_in_src:
                emitted = keys_in_src
        except Exception:
            pass

        missing = manifest_names - emitted
        extra = emitted - manifest_names
        if missing:
            print(f"[TextureSynth] CONTRACT FAIL: {cls.__name__} ({sv}) "
                  f"does NOT emit manifest param(s): {sorted(missing)}")
            issues += 1
        if extra:
            # Extras are warnings, not errors — they're harmless dead keys.
            print(f"[TextureSynth] contract note: {cls.__name__} ({sv}) "
                  f"emits keys not in manifest: {sorted(extra)} (ignored at runtime)")

    if issues == 0:
        print(f"[TextureSynth] Param contract: OK ({checked} nodes verified)")
    else:
        print(f"[TextureSynth] Param contract: {issues} FAIL(s) — sliders may misbehave.")


def register():
    for cls in _generated_classes:
        bpy.utils.register_class(cls)

    from ..core import cpp_module
    validate_named_param_contract(cpp_module)


def unregister():
    for cls in reversed(_generated_classes):
        bpy.utils.unregister_class(cls)