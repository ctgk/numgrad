import numpy as np

from numflow._decorators import differentiable
from numflow._variable import Variable


def _transpose_gradient(dy, _y, _x, axes=None):
    if axes is None:
        return np.transpose(dy)
    return np.transpose(dy, np.argsort(axes))


@differentiable(_transpose_gradient)
def _transpose(self: np.ndarray, axes=None):
    return self.transpose(axes)


Variable.transpose = lambda self, *axes: _transpose(
    self,
    *({0: tuple(), 1: axes}.get(len(axes), (axes,))),
)
Variable.transpose.__doc__ = np.ndarray.transpose.__doc__
Variable.T = property(lambda self: self.transpose())
