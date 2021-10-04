import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Square(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    @staticmethod
    def _forward_numpy(x):
        return np.square(x)

    @staticmethod
    def _backward_numpy(dy, x):
        return 2 * x * dy


@_typecheck(exclude_args=('x',))
def square(x: Array, *, name: str = None) -> Array:
    """Return square of each element.

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Square of each element

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.square([1, 2, -3])
    array([1., 4., 9.])
    """
    return _Square(x, name=name).forward()
