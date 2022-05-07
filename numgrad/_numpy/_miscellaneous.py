import numpy as np

from numgrad._decorators import _register_gradient


@_register_gradient(np.sqrt)
def _sqrt_gradient(doutput, output, _):
    return 0.5 / output * doutput


@_register_gradient(np.cbrt)
def _cbrt_gradient(dy, y, _x):
    return dy / (3 * y ** 2)


@_register_gradient(np.square)
def _square_gradient(dy, _, x):
    return 2 * x * dy


@_register_gradient(np.absolute)
def _absolute_gradient(dy, _y, x):
    return dy * np.sign(x)


@_register_gradient(np.fabs)
def _fabs_gradient(dy, _y, x):
    return dy * np.sign(x)
