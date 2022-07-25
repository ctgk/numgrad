# https://numpy.org/doc/stable/reference/arrays.ndarray.html

import numpy as np

from numgrad._variable import Variable
from numgrad._vjp import custom_vjp


def _getitem_vjp(g, r, x, key):
    dx = np.zeros_like(x)
    dx[key] = g
    return dx


Variable.__getitem__ = custom_vjp(_getitem_vjp)(
    lambda self, key: self[key])
Variable.__getitem__.__doc__ = np.ndarray.__getitem__.__doc__
