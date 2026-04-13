from __future__ import annotations

from functools import wraps

import warp as wp


def _is_cpu_device(device) -> bool:
    if device == "cpu":
        return True

    return bool(getattr(device, "is_cpu", False))


def _disable_packed_vbo_arrays(viewer) -> None:
    viewer._capsule_keys = set()
    viewer._packed_groups = []
    viewer._packed_write_indices = None
    viewer._packed_world_xforms = None
    viewer._packed_vbo_xforms = None
    viewer._packed_vbo_xforms_host = None


def enable_cpu_pinned_fallback(viewer) -> None:
    """Allow Newton's OpenGL viewer to run when Warp pinned CPU allocs fail.

    On some systems Warp drops back to CPU execution but still fails when a
    downstream viewer requests pinned host memory. In that case we retry the
    allocation without pinning so the OpenGL viewer can still start.
    """

    try:
        import newton._src.viewer.viewer_gl as viewer_gl  # noqa: PLC0415
    except Exception:
        return

    viewer_cls = getattr(viewer_gl, "ViewerGL", None)
    if viewer_cls is None or not isinstance(viewer, viewer_cls):
        return

    if getattr(viewer_cls, "_soma_cpu_pinned_fallback_enabled", False):
        return

    original_build = viewer_cls._build_packed_vbo_arrays

    @wraps(original_build)
    def _build_packed_vbo_arrays_with_fallback(self, *args, **kwargs):
        if not getattr(self.device, "is_cuda", False):
            _disable_packed_vbo_arrays(self)
            return

        original_empty = wp.empty

        def _empty_with_fallback(*empty_args, **empty_kwargs):
            wants_pinned_cpu = empty_kwargs.get("pinned", False) and _is_cpu_device(empty_kwargs.get("device"))
            if not wants_pinned_cpu:
                return original_empty(*empty_args, **empty_kwargs)

            try:
                return original_empty(*empty_args, **empty_kwargs)
            except Exception:
                fallback_kwargs = dict(empty_kwargs)
                fallback_kwargs["pinned"] = False
                return original_empty(*empty_args, **fallback_kwargs)

        wp.empty = _empty_with_fallback
        try:
            return original_build(self, *args, **kwargs)
        finally:
            wp.empty = original_empty

    viewer_cls._build_packed_vbo_arrays = _build_packed_vbo_arrays_with_fallback
    viewer_cls._soma_cpu_pinned_fallback_enabled = True
