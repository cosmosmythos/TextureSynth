"""Evaluation Loop: manages the event-driven updates (msgbus) and minimal timer polling.
Drains Vulkan compilation fence signals and GPU readbacks periodically."""
import bpy
from . import cpp_module
from . import engine_bridge
from . import logging as _tslog

_topology_dirty = False
_params_dirty   = False
_compiling      = False
_force_render   = False
_last_active_node_id = None

_msgbus_owner = object()


def request_topology_update():
    global _topology_dirty
    _topology_dirty = True


def request_param_update():
    global _params_dirty
    _params_dirty = True


def _find_ts_trees():
    """Yield all TextureSynth node trees in the current .blend."""
    for ng in bpy.data.node_groups:
        if getattr(ng, "bl_idname", None) == "TextureSynthTreeType":
            yield ng


def _on_node_select_change():
    """Callback fired on node selection change. Detects active node changes and requests re-render."""
    engine = cpp_module.get_engine()
    if engine is None or not engine.has_pipeline():
        return
    for tree in _find_ts_trees():
        active = getattr(tree.nodes, "active", None)
        if active is None or not hasattr(active, "stable_id"):
            continue
        if getattr(active, "sv_type", None) is None:
            continue
        try:
            new_id = int(active.stable_id())
        except (TypeError, ValueError):
            continue
        if new_id == engine_bridge._last_active_node_id:
            continue
        
        # Try to set active node in engine. If the graph doesn't have it yet
        # (e.g. newly added node), request a full rebuild from the node tree.
        if engine_bridge._check_active_node_change(tree, engine):
            try:
                engine_bridge.update_params_only(force_submit=True)
            except Exception as e:
                _tslog.error(f"active-node dispatch exception: {e}")
        else:
            # set_active_node failed — the engine's cached graph doesn't have this
            # node yet. Request a full rebuild from the Blender node tree.
            request_topology_update()


def _subscribe_msgbus():
    """Subscribe to node select changes to trigger preview updates on selection."""
    bpy.msgbus.subscribe_rna(
        key=(bpy.types.Node, "select"),
        owner=_msgbus_owner,
        args=(),
        notify=_on_node_select_change,
    )


def _load_post_handler(dummy):
    """Re-subscribe msgbus after .blend load (msgbus is cleared on load)."""
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    _subscribe_msgbus()


def _evaluation_timer():
    global _topology_dirty, _params_dirty, _compiling, _force_render
    global _last_active_node_id

    if not cpp_module.is_loaded():
        return 1.0
    engine = cpp_module.get_engine()
    if engine is None:
        return 1.0

    # Poll Vulkan async compile queue.
    try:
        engine.poll_pending_compiles()
    except Exception as e:
        _tslog.error(f"poll exception: {e}")

    # Sync compile errors to node UI.
    try:
        tree = engine_bridge._find_node_tree()
        if tree:
            engine_bridge.sync_node_errors(tree)
    except Exception as e:
        _tslog.error(f"sync errors exception: {e}")

    # Poll for active-node change (msgbus on Nodes.active is unreliable in 4.2+).
    try:
        if tree is not None and engine_bridge._check_active_node_change(tree, engine):
            engine_bridge.update_params_only(force_submit=True)
    except Exception as e:
        _tslog.error(f"active-node poll exception: {e}")

    # Resubmit graph if topology changed.
    if _topology_dirty:
        _topology_dirty = False
        generation = engine_bridge.submit_graph()
        if generation:
            _params_dirty = False
            _force_render = True
            _compiling = not engine.is_generation_ready(generation)
            _last_active_node_id = None
        return 0.01

    submitted = engine_bridge.submitted_generation()
    ready = bool(submitted and engine.is_generation_ready(submitted))

    # Check if compiling is finished — force a render dispatch so the new graph
    # produces its first frame (e.g. after set_active_node triggered async compile).
    if _compiling and ready:
        _compiling = False
        _force_render = True

    # Render if graph is ready and params or topology changed.
    needs_dispatch = ready and (_force_render or _params_dirty)

    if needs_dispatch:
        try:
            state = engine_bridge.update_params_only(force_submit=True)
        except Exception as e:
            _tslog.error(f"dispatch exception: {e}")
            return 0.1

        if state == 'landed':
            _force_render = False
            _params_dirty = False
        return 0.05

    # Sleep less if still compiling shaders.
    if _compiling:
        return 0.05

    # Poll for readback if frames are in flight.
    if engine.async_in_flight():
        try:
            engine_bridge.update_params_only(force_submit=False)
        except Exception as e:
            _tslog.error(f"idle poll exception: {e}")
        return 0.05

    return 0.1


def register():
    if not bpy.app.timers.is_registered(_evaluation_timer):
        bpy.app.timers.register(_evaluation_timer,
                                first_interval=0.5, persistent=True)
    if _load_post_handler not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_load_post_handler)
    _subscribe_msgbus()


def unregister():
    global _topology_dirty, _params_dirty, _compiling, _force_render
    global _last_active_node_id
    _topology_dirty = _params_dirty = _compiling = _force_render = False
    _last_active_node_id = None
    engine_bridge._last_active_fingerprint = None
    engine_bridge._submitted_generation = 0
    engine_bridge._last_applied_generation = 0
    engine_bridge._last_pushed_param_hash = None
    engine_bridge._last_active_node_id = None
    if bpy.app.timers.is_registered(_evaluation_timer):
        bpy.app.timers.unregister(_evaluation_timer)
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post_handler)
