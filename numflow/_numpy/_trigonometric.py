"""Trigonometric functions.

https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
"""

import numpy as np

from numflow._decorators import _register_gradient
from numflow._utils._unbroadcast import _unbroadcast_to


@_register_gradient(np.sin)
def _sin_gradient(doutput, output, x):
    return np.cos(x) * doutput


@_register_gradient(np.cos)
def _cos_gradient(doutput, output, x):
    return -np.sin(x) * doutput


@_register_gradient(np.tan)
def _tan_gradient(doutput, output, x):
    return (1 + np.square(output)) * doutput


@_register_gradient(np.arcsin)
def _arcsin_gradient(dy, y, x):
    return dy / np.cos(y)


@_register_gradient(np.arccos)
def _arccos_gradient(dy, y, x):
    return -dy / np.sin(y)


@_register_gradient(np.arctan)
def _arctan_gradient(dy, y, x):
    return (np.cos(y) ** 2) * dy


@_register_gradient(np.hypot)
def _hypot_gradient(do, o, x, y):
    return (
        _unbroadcast_to(x / o, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(y / o, y.shape) if hasattr(y, 'shape') else None,
    )
