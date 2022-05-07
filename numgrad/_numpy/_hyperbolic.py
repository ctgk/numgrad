"""Hyperbolic functions.

https://numpy.org/doc/stable/reference/routines.math.html#hyperbolic-functions
"""


import numpy as np

from numgrad._decorators import _register_gradient


@_register_gradient(np.sinh)
def _sinh_gradient(doutput, _output, x):
    return np.cosh(x) * doutput


@_register_gradient(np.cosh)
def _cosh_gradient(doutput, _output, x):
    return np.sinh(x) * doutput


@_register_gradient(np.tanh)
def _tanh_gradient(doutput, output, _x):
    return (1 - np.square(output)) * doutput


@_register_gradient(np.arcsinh)
def _arcsinh_gradient(dy, y, _x):
    return dy / np.cosh(y)


@_register_gradient(np.arccosh)
def _arccosh_gradient(dy, y, _x):
    return dy / np.sinh(y)


@_register_gradient(np.arctanh)
def _arctanh_gradient(dy, _y, x):
    return dy / (1 - x ** 2)
