"""Trigonometric functions.

https://numpy.org/doc/stable/reference/routines.math.html#trigonometric-functions
"""

import numpy as np

from numgrad._decorators import _register_gradient
from numgrad._utils._unbroadcast import _unbroadcast_to


@_register_gradient(np.sin)
def _sin_gradient(doutput, _output, x):
    """Gradient of np.sin ufunc."""
    return np.cos(x) * doutput


@_register_gradient(np.cos)
def _cos_gradient(doutput, _output, x):
    """Gradient of np.cos ufunc."""
    return -np.sin(x) * doutput


@_register_gradient(np.tan)
def _tan_gradient(doutput, output, _x):
    """Gradient of np.tan ufunc."""
    return (1 + np.square(output)) * doutput


@_register_gradient(np.arcsin)
def _arcsin_gradient(dy, y, _x):
    """Gradient of np.arcsin ufunc."""
    return dy / np.cos(y)


@_register_gradient(np.arccos)
def _arccos_gradient(dy, y, _x):
    """Gradient of np.arcsin ufunc."""
    return -dy / np.sin(y)


@_register_gradient(np.arctan)
def _arctan_gradient(dy, y, _x):
    """Gradient of np.arctan ufunc."""
    return (np.cos(y) ** 2) * dy


@_register_gradient(np.hypot)
def _hypot_gradient(do, o, x, y):
    """Gradient of np.hypot ufunc."""
    return (
        _unbroadcast_to(do * x / o, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(do * y / o, y.shape) if hasattr(y, 'shape') else None,
    )
