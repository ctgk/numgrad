"""Extrema finding.

https://numpy.org/doc/stable/reference/routines.math.html#extrema-finding
"""

import numpy as np

from numgrad._decorators import _register_gradient
from numgrad._utils._unbroadcast import _unbroadcast_to


@_register_gradient(np.maximum)
def _maximum_gradient(dz, z, x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 0 and y.ndim == 0:
        return (dz, 0) if x > y else (0, dz)
    dx_broadcasted, dy_broadcasted = np.copy(dz), np.copy(dz)
    dx_broadcasted[np.where(x != z)] = 0
    dy_broadcasted[np.where(y != z)] = 0
    return (
        _unbroadcast_to(dx_broadcasted, x.shape),
        _unbroadcast_to(dy_broadcasted, y.shape),
    )


@_register_gradient(np.fmax)
def _fmax_gradient(dz, z, x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 0 and y.ndim == 0:
        return (dz, 0) if (np.isnan(y) or x > y) else (0, dz)
    dx_broadcasted, dy_broadcasted = np.copy(dz), np.copy(dz)
    dx_broadcasted[np.where(x != z)] = 0
    dy_broadcasted[np.where(y != z)] = 0
    return (
        _unbroadcast_to(dx_broadcasted, x.shape),
        _unbroadcast_to(dy_broadcasted, y.shape),
    )


@_register_gradient(np.amax)
def _max_gradient(doutput, _, x, axis=None, keepdims=False, **kwargs):
    """Gradient of np.max which supports __array_function__."""
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


@_register_gradient(np.nanmax)
def _nanmax_gradient(dy, y, x, axis=None, keepdims=False):
    """Gradient of np.nanmax which supports __array_function__."""
    if x.ndim == 0:
        return np.nan if np.isnan(x) else dy
    if all((
        isinstance(dy, np.ndarray),
        (not keepdims),
        (axis is not None),
    )):
        axis_positive = []
        for ax in axis if isinstance(axis, tuple) else (axis,):
            axis_positive.append(x.ndim + ax if ax < 0 else ax)
        for ax in sorted(axis_positive):
            dy = np.expand_dims(dy, ax)
            y = np.expand_dims(y, ax)
    dx = 1 * np.broadcast_to(dy, x.shape)
    dx[np.where(x != y)] = 0
    return dx


@_register_gradient(np.minimum)
def _minimum_gradient(dz, z, x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 0 and y.ndim == 0:
        return (dz, 0) if x < y else (0, dz)
    dx_broadcasted, dy_broadcasted = np.copy(dz), np.copy(dz)
    dx_broadcasted[np.where(x != z)] = 0
    dy_broadcasted[np.where(y != z)] = 0
    return (
        _unbroadcast_to(dx_broadcasted, x.shape),
        _unbroadcast_to(dy_broadcasted, y.shape),
    )


@_register_gradient(np.fmin)
def _fmin_gradient(dz, z, x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim == 0 and y.ndim == 0:
        return (dz, 0) if (np.isnan(y) or x < y) else (0, dz)
    dx_broadcasted, dy_broadcasted = np.copy(dz), np.copy(dz)
    dx_broadcasted[np.where(x != z)] = 0
    dy_broadcasted[np.where(y != z)] = 0
    return (
        _unbroadcast_to(dx_broadcasted, x.shape),
        _unbroadcast_to(dy_broadcasted, y.shape),
    )


@_register_gradient(np.amin)
def _min_gradient(doutput, _, x, axis=None, keepdims=False, **kwargs):
    """Gradient of np.min which supports __array_function__."""
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


@_register_gradient(np.nanmin)
def _nanmin_gradient(dy, y, x, axis=None, keepdims=False):
    """Gradient of np.nanmin which supports __array_function__."""
    if x.ndim == 0:
        return np.nan if np.isnan(x) else dy
    if all((
        isinstance(dy, np.ndarray),
        (not keepdims),
        (axis is not None),
    )):
        axis_positive = []
        for ax in axis if isinstance(axis, tuple) else (axis,):
            axis_positive.append(x.ndim + ax if ax < 0 else ax)
        for ax in sorted(axis_positive):
            dy = np.expand_dims(dy, ax)
            y = np.expand_dims(y, ax)
    dx = 1 * np.broadcast_to(dy, x.shape)
    dx[np.where(x != y)] = 0
    return dx
