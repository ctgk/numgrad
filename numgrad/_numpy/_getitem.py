import numpy as np

from numgrad._decorators import differentiable
from numgrad._variable import Variable


def _getitem_gradient(dy, _y, x, key):
    dx = np.zeros_like(x)
    dx[key] = dy
    return dx


@differentiable(_getitem_gradient)
def _getitem(self, key):
    return self[key]


Variable.__getitem__ = _getitem
Variable.__getitem__.__doc__ = np.ndarray.__getitem__.__doc__
