import numpy as np

from numgrad._decorators import _register_gradient
from numgrad._numpy import (  # noqa: F401
    _arithmetic,
    _exp_log,
    _extrema_finding,
    _getitem,
    _hyperbolic,
    _miscellaneous,
    _random,
    _reshape,
    _statistics,
    _transpose,
    _trigonometric,
)
from numgrad._utils._unbroadcast import _unbroadcast_to


@_register_gradient(np.matmul)
def _matmul_gradient(do, _, x, y):
    """Gradient of np.matmul ufunc."""
    x, y = np.asarray(x), np.asarray(y)
    if y.ndim == 1:
        do = np.expand_dims(do, -1)
        do = np.broadcast_to(do, x.shape)
        dx = do * y
        dy = _unbroadcast_to(do * x, y.shape)
        return dx, dy
    if x.ndim == 1:
        do = np.expand_dims(do, -2)
        do = np.broadcast_to(do, y.shape)
        dx = _unbroadcast_to((do * y).sum(axis=-1), x.shape)
        dy = do * x[:, None]
        return dx, dy
    if x.ndim == y.ndim == 2:
        dx = do @ y.T
        dy = x.T @ do
        return dx, dy
    else:
        dx = _unbroadcast_to(do @ np.swapaxes(y, -1, -2), x.shape)
        dy = _unbroadcast_to(np.swapaxes(x, -1, -2) @ do, y.shape)
        return dx, dy


@_register_gradient(np.square)
def _square_gradient(dy, _, x):
    return 2 * x * dy


@_register_gradient(np.sqrt)
def _sqrt_gradient(doutput, output, _):
    return 0.5 / output * doutput


@_register_gradient(np.mean)
def _mean_gradient(doutput, _, x, axis=None, keepdims=False):
    """Gradient of np.mean which supports __array_function__."""
    if all((
        isinstance(doutput, np.ndarray),
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
            doutput = np.expand_dims(doutput, ax)
    dx = np.broadcast_to(doutput, x.shape)
    dx = dx * doutput.size / x.size
    return dx


@_register_gradient(np.sum)
def _sum_gradient(doutput, _, x, axis=None, keepdims=False, **kwargs):
    """Gradient of np.sum which supports __array_function__."""
    if all((
        isinstance(doutput, np.ndarray),
        (not keepdims),
        (axis is not None),
    )):
        axis_positive = []
        if isinstance(axis, int):
            axis = (axis,)
        for ax in axis:
            if ax < 0:
                axis_positive.append(x.ndim + ax)
            else:
                axis_positive.append(ax)
        for ax in sorted(axis_positive):
            doutput = np.expand_dims(doutput, ax)
    dx = np.broadcast_to(doutput, x.shape)
    return dx


__all__ = []
