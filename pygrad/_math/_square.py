import numpy as np

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args


class _Square(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    @staticmethod
    def _forward_numpy(x):
        return np.square(x)

    @staticmethod
    def _backward_numpy(dy, x):
        return 2 * x * dy


@_typecheck_args
def square(x, *, name: str = None) -> Array:
    """Return square of each element

    Parameters
    ----------
    x
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Square of each element

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.square([1, 2, -3])
    array([1, 4, 9])
    """
    return _Square(x, name=name).forward()
