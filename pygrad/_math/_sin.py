import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Sin(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        return np.sin(x)

    def _backward_numpy(self, dy, x):
        return dy * np.cos(x)


@_typecheck(exclude_args=('x',))
def sin(x: Array, *, name: str = None) -> Array:
    """Return trigonometric sine of each element

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Trigonometric sine of each element

    Examples
    --------
    >>> import pygrad as gd; from math import pi
    >>> gd.sin([0, -pi / 6, 17 * pi / 6, -pi / 2])
    array([ 0. , -0.5,  0.5, -1. ])
    """
    return _Sin(x, name=name).forward()
