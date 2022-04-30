import numpy as np

from pygrad._decorators import differentiable
from pygrad._variable import Variable


def _getitem_gradient(dy, y, x, key):
    dx = np.zeros_like(x)
    dx[key] = dy
    return dx


@differentiable(_getitem_gradient)
def _getitem(self, key):
    return self[key]


Variable.__getitem__ = _getitem
Variable.__getitem__.__doc__ = np.ndarray.__getitem__.__doc__
