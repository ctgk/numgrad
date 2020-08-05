import numpy as np

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args
from pygrad._utils._unbroadcast import _unbroadcast_to


class _Subtract(_Operator):

    def __init__(self, x: Array, y: Array, name: str = None):
        super().__init__(x, y, name=name)

    @staticmethod
    def _forward_numpy(x, y):
        return x - y

    @staticmethod
    def _backward_numpy(delta: np.ndarray, x: np.ndarray, y: np.ndarray):
        dx = _unbroadcast_to(delta, x.shape)
        dy = -_unbroadcast_to(delta, y.shape)
        return dx, dy


@_typecheck_args
def subtract(x, y, name: str = None) -> Array:
    """Return element-wise subtraction of two arrays.

    Parameters
    ----------
    x
        Input array.
    y
        Another input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Element-wise subtraction of two arrays.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.subtract([[1, 2], [2, 3]], [-1, 3])
    array([[ 2, -1],
           [ 3,  0]])
    """
    return _Subtract(x, y, name=name).forward()
