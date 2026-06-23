"""Shared utils for specialized node modules."""


def update_param(self, context):
    """Route the change into the param-only GPU dispatch path (no shader recompile)."""
    try:
        from ...core.evaluation import request_param_update
        request_param_update()
    except Exception:
        pass
