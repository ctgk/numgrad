import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Cos(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    @staticmethod
    def _forward_numpy(x):
        return np.cos(x)

    def _backward_numpy(self, dy, x):
        return -dy * np.sin(x)


@_typecheck(exclude=('x',))
def cos(x: Array, *, name: str = None) -> Array:
    """Return trigonometric cosine of each element.

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Trigonometric cosine of each element

    Examples
    --------
    >>> import pygrad as pg; from math import pi
    >>> pg.cos([0, pi / 3, 14 * pi / 3])
    array([ 1. ,  0.5, -0.5])
    """
    return _Cos(x, name=name).forward()
