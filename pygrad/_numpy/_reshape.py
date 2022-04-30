import numpy as np

from pygrad._decorators import differentiable
from pygrad._variable import Variable


def _reshape_gradient(dy, y, x, newshape, order=None):
    return dy.reshape(*x.shape, order=order)


@differentiable(_reshape_gradient)
def _reshape(self: np.ndarray, newshape, order=None):
    return self.reshape(*newshape, order=order)


Variable.reshape = lambda self, *args, **kwargs: _reshape(
    self, *(args if len(args) == 1 else (args,)), **kwargs)
Variable.reshape.__doc__ = np.ndarray.reshape.__doc__
