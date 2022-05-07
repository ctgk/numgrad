import numpy as np

from numgrad._decorators import _register_gradient


@_register_gradient(np.nanmean)
def _nanmean_gradient(dy, _y, x, axis=None, *, keepdims=False):
    """Gradient of nanmean which uses __array_function__."""
    if all((
        isinstance(dy, np.ndarray),
        (not keepdims),
        (axis is not None),
    )):
        axis_positive = []
        for ax in axis if isinstance(axis, tuple) else (axis,):
            if ax < 0:
                axis_positive.append(x.ndim + ax)
            else:
                axis_positive.append(ax)
        for ax in sorted(axis_positive):
            dy = np.expand_dims(dy, ax)
    nan_mask = np.isnan(x)
    dx = np.asarray(np.broadcast_to(dy, x.shape) / np.sum(
        ~nan_mask, axis=axis, keepdims=True))
    dx[nan_mask] = 0
    return dx
