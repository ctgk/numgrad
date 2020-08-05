import numpy as np

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args
from pygrad._utils._unbroadcast import _unbroadcast_to


class _Multiply(_Operator):

    def __init__(self, x: Array, y: Array, name: str = None):
        super().__init__(x, y, name=name)

    @staticmethod
    def _forward_numpy(x, y):
        return x * y

    @staticmethod
    def _backward_numpy(delta: np.ndarray, x: np.ndarray, y: np.ndarray):
        dx = _unbroadcast_to(delta * y, x.shape)
        dy = _unbroadcast_to(delta * x, y.shape)
        return dx, dy


@_typecheck_args
def multiply(x, y, name: str = None) -> Array:
    """Return element-wise multiplication of two arrays.

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
        Element-wise multiplication of two arrays.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.multiply([[1, 2], [2, 3]], [-1, 3])
    array([[-1,  6],
           [-2,  9]])
    """
    return _Multiply(x, y, name=name).forward()
