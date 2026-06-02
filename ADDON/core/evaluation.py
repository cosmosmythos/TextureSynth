"""
Evaluation Loop — timer-based update system.

State semantics (all booleans are "user intent" sticky flags):
  _topology_dirty : graph structure changed; needs resubmit
  _params_dirty   : only slider values changed; just re-dispatch
  _compiling      : a compile is in flight (informational)
  _force_render   : render must happen on the next tick where
                    has_pipeline() is True. NEVER cleared except
                    by a successful render or by unregister().

Invariant:
  • A failed submit (transient: no output link, no tree) does NOT
    clear _force_render. The user's last successful submit may still
    be compiling; when it lands we must still render it.
  • A submit_graph() success implicitly satisfies _force_render's
    intent for the previous topology — but we keep it set because
    the new submit's render is also pending.
"""
import bpy
from . import cpp_module
from . import engine_bridge
from . import logging as _tslog

_topology_dirty = False
_params_dirty   = False
_compiling      = False
_force_render   = False


def request_topology_update():
    global _topology_dirty
    _topology_dirty = True


def request_param_update():
    global _params_dirty
    _params_dirty = True


def _evaluation_timer():
    global _topology_dirty, _params_dirty, _compiling, _force_render

    if not cpp_module.is_loaded():
        return 1.0
    engine = cpp_module.get_engine()
    if engine is None:
        return 1.0

    # 1. Always poll compile queue.
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

    # 2. Topology change → resubmit.
    if _topology_dirty:
        _topology_dirty = False
        generation = engine_bridge.submit_graph()
        if generation:
            _params_dirty = False
            _force_render = True
            _compiling = not engine.is_generation_ready(generation)
        # else: transient invalid — keep _force_render as-is.
        return 0.25

    submitted = engine_bridge.submitted_generation()
    ready = bool(submitted and engine.is_generation_ready(submitted))

    # 3. Compile-finished detection.
    if _compiling and ready:
        _compiling = False

    # 4. Decide whether to dispatch this tick.
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
        # 'in_flight' or 'idle' → keep ticking; don't clear flags.
        return 0.05

    # 5. Still compiling — fast tick.
    if _compiling:
        return 0.05

    # 6. Idle but ring may have a straggler. Poll-only, no submit.
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


def unregister():
    global _topology_dirty, _params_dirty, _compiling, _force_render
    _topology_dirty = _params_dirty = _compiling = _force_render = False
    # Reset bridge state too so a re-register starts clean.
    engine_bridge._last_active_fingerprint = None
    engine_bridge._submitted_generation = 0
    engine_bridge._last_applied_generation = 0
    engine_bridge._last_pushed_param_hash = None
    if bpy.app.timers.is_registered(_evaluation_timer):
        bpy.app.timers.unregister(_evaluation_timer)
