"""Exponents and logarithms.

https://numpy.org/doc/stable/reference/routines.math.html#exponents-and-logarithms
"""

import numpy as np

from numflow._decorators import _register_gradient
from numflow._utils._unbroadcast import _unbroadcast_to


@_register_gradient(np.exp)
def _exp_gradient(doutput, output, x):
    return output * doutput


@_register_gradient(np.expm1)
def _expm1_gradient(dy, y, x):
    return (y + 1) * dy


@_register_gradient(np.exp2)
def _exp2_gradient(dy, y, x):
    return y * np.log(2) * dy


@_register_gradient(np.log)
def _log_gradient(doutput, output, x):
    return doutput / x


@_register_gradient(np.log10)
def _log10_gradient(dy, y, x):
    return dy / (x * np.log(10))


@_register_gradient(np.log2)
def _log2_gradient(dy, y, x):
    return dy / (x * np.log(2))


@_register_gradient(np.log1p)
def _log1p_gradient(dy, y, x):
    return dy / (1 + x)


@_register_gradient(np.logaddexp)
def _logaddexp_gradient(dz, z, x, y):
    return (
        _unbroadcast_to(
            np.exp(x - z) * dz, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(
            np.exp(y - z) * dz, y.shape) if hasattr(y, 'shape') else None,
    )


@_register_gradient(np.logaddexp2)
def _logaddexp2_gradient(dz, z, x, y):
    return (
        _unbroadcast_to(
            np.exp2(x - z) * dz, x.shape) if hasattr(x, 'shape') else None,
        _unbroadcast_to(
            np.exp2(y - z) * dz, y.shape) if hasattr(y, 'shape') else None,
    )
