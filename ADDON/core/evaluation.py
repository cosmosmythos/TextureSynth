"""Event-driven update loop: msgbus subscriptions + timer that drains Vulkan compile fences and GPU readbacks."""
import bpy
from . import cpp_module
from . import engine_bridge
from . import logging as _tslog

_topology_dirty = False
_params_dirty   = False
_compiling      = False
_force_render   = False

_msgbus_owner = object()


def request_topology_update():
    global _topology_dirty
    import traceback
    print(f"[DIAG] request_topology_update called:")
    traceback.print_stack(limit=8)
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
    """Node-select callback: detect active-node change and request re-render."""
    global _compiling, _force_render
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
        _tslog.info(f"[DIAG] select_change: new_id={new_id} prev_id={engine_bridge._last_active_node_id}")
        print(f"[DIAG] select_change: new_id={new_id} prev_id={engine_bridge._last_active_node_id}")

        # If the engine's cached graph doesn't have this node yet, request a full rebuild.
        if engine_bridge._check_active_node_change(tree, engine):
            print(f"[DIAG] select_change: _check_active_node_change=True, dispatched")
            gen = engine_bridge._submitted_generation
            if gen and not engine.is_generation_ready(gen):
                _compiling = True
                _force_render = True
            else:
                _force_render = True
            try:
                engine_bridge.update_params_only(force_submit=True)
            except Exception as e:
                _tslog.error(f"active-node dispatch exception: {e}")
        else:
            print(f"[DIAG] select_change: _check_active_node_change=False, requesting topology rebuild")
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

    if not cpp_module.is_loaded():
        return 1.0
    engine = cpp_module.get_engine()
    if engine is None:
        return 1.0

    try:
        engine.poll_pending_compiles()
    except Exception as e:
        _tslog.error(f"poll exception: {e}")

    tree = engine_bridge._find_node_tree()

    try:
        if tree:
            engine_bridge.sync_node_errors(tree)
    except Exception as e:
        _tslog.error(f"sync errors exception: {e}")

    # Blender 5.0 may not fire tree.update() on mute — poll for mute-state changes.
    try:
        if tree is not None and engine_bridge.poll_mute_state(tree):
            request_topology_update()
    except Exception as e:
        _tslog.error(f"mute poll exception: {e}")

    # msgbus on Nodes.active is unreliable in 4.2+ — poll for active-node change.
    try:
        if tree is not None:
            if engine_bridge._check_active_node_change(tree, engine):
                _tslog.debug(f"active-node changed, gen={engine_bridge._submitted_generation}")
                gen = engine_bridge._submitted_generation
                if gen and not engine.is_generation_ready(gen):
                    _compiling = True
                    _force_render = True
                else:
                    _force_render = True
                engine_bridge.update_params_only(force_submit=True)
            else:
                # set_active_node failed (node not in engine's subgraph):
                # only rebuild if the active node actually changed.
                active = tree.nodes.active
                if (active is not None
                        and hasattr(active, "stable_id")
                        and int(active.stable_id()) != engine_bridge._last_active_node_id):
                    _tslog.debug(f"active-node not in engine, requesting rebuild "
                                 f"(active_id={int(active.stable_id())} "
                                 f"last_id={engine_bridge._last_active_node_id})")
                    request_topology_update()
    except Exception as e:
        _tslog.error(f"active-node poll exception: {e}")

    # Topology fingerprint catches link changes missed by tree.update().
    try:
        if tree is not None and engine_bridge._submitted_generation:
            fp = engine_bridge._active_subgraph_fingerprint(tree)
            if fp != engine_bridge._last_active_fingerprint:
                print(f"[DIAG] timer: fingerprint changed, requesting topology rebuild")
                request_topology_update()
    except Exception as e:
        _tslog.error(f"topology fingerprint poll exception: {e}")

    # Resubmit graph if topology changed.
    if _topology_dirty:
        _topology_dirty = False
        print(f"[DIAG] timer: _topology_dirty=True, calling submit_graph()")
        generation = engine_bridge.submit_graph()
        if generation:
            _params_dirty = False
            _force_render = True
            _compiling = not engine.is_generation_ready(generation)
            print(f"[DIAG] timer: submit_graph returned gen={generation} compiling={_compiling}")
            # Sync engine_bridge's active-node tracking so _check_active_node_change()
            # won't trigger another request_topology_update() on the next tick, which
            # would resubmit before compilation finishes.
            active = tree.nodes.active if tree else None
            if (active is not None
                    and hasattr(active, 'stable_id')
                    and getattr(active, 'sv_type', None) is not None):
                engine_bridge._last_active_node_id = int(active.stable_id())
        return 0.01

    submitted = engine_bridge.submitted_generation()
    ready = bool(submitted and engine.is_generation_ready(submitted))

    # Compile finished — force a render dispatch so the new graph emits its first frame
    # (e.g. after set_active_node triggered async compile).
    if _compiling and ready:
        _compiling = False
        _force_render = True

    needs_dispatch = ready and (_force_render or _params_dirty)

    if needs_dispatch:
        try:
            state = engine_bridge.update_params_only(force_submit=True)
            _tslog.debug(f"dispatch state={state}")
        except Exception as e:
            _tslog.error(f"dispatch exception: {e}")
            return 0.1

        if state == 'landed':
            _force_render = False
            _params_dirty = False
        return 0.05

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
    _topology_dirty = _params_dirty = _compiling = _force_render = False
    engine_bridge._last_active_fingerprint = None
    engine_bridge._submitted_generation = 0
    engine_bridge._last_applied_generation = 0
    engine_bridge._last_pushed_param_hash = None
    engine_bridge._last_active_node_id = None
    engine_bridge._last_mute_snapshot = {}
    engine_bridge._image_cache.clear()
    engine_bridge._pending_image_uploads.clear()
    if bpy.app.timers.is_registered(_evaluation_timer):
        bpy.app.timers.unregister(_evaluation_timer)
    bpy.msgbus.clear_by_owner(_msgbus_owner)
    if _load_post_handler in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post_handler)
