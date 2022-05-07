"""Arithmetic operations.

https://numpy.org/doc/stable/reference/routines.math.html#arithmetic-operations
"""

import numpy as np

from numgrad._decorators import _register_gradient
from numgrad._utils._unbroadcast import _unbroadcast_to


@_register_gradient(np.add)
def _add_gradient(doutput, _, x, y):
    return (
        _unbroadcast_to(doutput, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(doutput, y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.reciprocal)
def _reciprocal_gradient(dy, y, _x):
    return dy * -(y ** 2)


@_register_gradient(np.positive)
def _positive_gradient(doutput, _do, _x):
    return doutput


@_register_gradient(np.negative)
def _negative_gradient(doutput, _do, _x):
    return -doutput


@_register_gradient(np.multiply)
def _multiply_gradient(doutput, _, x, y):
    return (
        _unbroadcast_to(doutput * y, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(doutput * x, y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.divide)
def _divide_gradient(do, _, x, y):
    return (
        _unbroadcast_to(do / y, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(
            -do * x / (y ** 2), y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.power)
def _power_gradient(dz, z, x, y):
    x, y = np.asarray(x), np.asarray(y)
    dx = _unbroadcast_to(y * np.power(x, y - 1) * dz, x.shape)
    dy = _unbroadcast_to(
        z * np.log(x) * dz, y.shape) if np.all(x > 0) else None
    return dx, dy


@_register_gradient(np.subtract)
def _subtract_gradient(doutput, _, x, y):
    return (
        _unbroadcast_to(doutput, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(-doutput, y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.floor_divide)
def _floor_divide_gradient(*args, **kwargs):
    raise NotImplementedError


@_register_gradient(np.float_power)
def _float_power_gradient(dz, z, x, y):
    x, y = np.asarray(x), np.asarray(y)
    dx = _unbroadcast_to(y * np.power(x, y - 1) * dz, x.shape)
    dy = _unbroadcast_to(
        z * np.log(x) * dz, y.shape) if np.all(x > 0) else None
    return dx, dy


@_register_gradient(np.fmod)
def _fmod_gradient(*args, **kwargs):
    raise NotImplementedError


@_register_gradient(np.mod)
def _mod_gradient(*args, **kwargs):
    raise NotImplementedError


@_register_gradient(np.modf)
def _modf_gradient(*args, **kwargs):
    raise NotImplementedError


@_register_gradient(np.remainder)
def _remainder_gradient(*args, **kwargs):
    raise NotImplementedError


@_register_gradient(np.divmod)
def _divmod_gradient(*args, **kwargs):
    raise NotImplementedError
