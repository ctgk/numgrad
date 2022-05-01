import numpy as np

from numflow._decorators import _register_gradient
from numflow._numpy import (  # noqa: F401
    _getitem, _hyperbolic, _random, _reshape, _transpose, _trigonometric,
)
from numflow._utils._unbroadcast import _unbroadcast_to


@_register_gradient(np.positive)
def _positive_gradient(doutput, output, x):
    return doutput


@_register_gradient(np.negative)
def _negative_gradient(doutput, output, x):
    return -doutput


@_register_gradient(np.add)
def _add_gradient(doutput, output, x, y):
    return (
        _unbroadcast_to(doutput, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(doutput, y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.subtract)
def _subtract_gradient(doutput, output, x, y):
    return (
        _unbroadcast_to(doutput, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(-doutput, y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.multiply)
def _multiply_gradient(doutput, output, x, y):
    return (
        _unbroadcast_to(doutput * y, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(doutput * x, y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.divide)
def _divide_gradient(do, o, x, y):
    return (
        _unbroadcast_to(do / y, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(
            -do * x / (y ** 2), y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.matmul)
def _matmul_gradient(do, o, x, y):
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
def _square_gradient(dy, y, x):
    return 2 * x * dy


@_register_gradient(np.sqrt)
def _sqrt_gradient(doutput, output, x):
    return 0.5 / output * doutput


@_register_gradient(np.exp)
def _exp_gradient(doutput, output, x):
    return output * doutput


@_register_gradient(np.log)
def _log_gradient(doutput, output, x):
    return doutput / x


@_register_gradient(np.maximum, method='reduce')
def _max_gradient(doutput, output, x, axis=None, keepdims=False, **kwargs):
    if x.ndim == 0:
        return doutput
    if all((
        isinstance(doutput, np.ndarray),
        (not keepdims),
        (axis is not None),
    )):
        axis_positive = []
        for ax in axis if isinstance(axis, tuple) else (axis,):
            axis_positive.append(x.ndim + ax if ax < 0 else ax)
        for ax in sorted(axis_positive):
            doutput = np.expand_dims(doutput, ax)
    dx = 1 * np.broadcast_to(doutput, x.shape)
    dx[np.where(x != x.max(axis=axis, keepdims=True))] = 0
    return dx


@_register_gradient(np.minimum, method='reduce')
def _min_gradient(doutput, output, x, axis=None, keepdims=False, **kwargs):
    if x.ndim == 0:
        return doutput
    if all((
        isinstance(doutput, np.ndarray),
        (not keepdims),
        (axis is not None),
    )):
        axis_positive = []
        for ax in axis if isinstance(axis, tuple) else (axis,):
            axis_positive.append(x.ndim + ax if ax < 0 else ax)
        for ax in sorted(axis_positive):
            doutput = np.expand_dims(doutput, ax)
    dx = 1 * np.broadcast_to(doutput, x.shape)
    dx[np.where(x != x.min(axis=axis, keepdims=True))] = 0
    return dx


@_register_gradient(np.mean)
def _mean_gradient(doutput, output, x, axis=None, keepdims=False):
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


@_register_gradient(np.add, method='reduce')
def _sum_gradient(doutput, output, x, axis=None, keepdims=False, **kwargs):
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