import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _Divide(_Operator):

    def __init__(self, x: Array, y: Array, name: str = None):
        super().__init__(x, y, name=name)

    @staticmethod
    def _forward_numpy(x, y):
        return x / y

    def _backward_numpy(self, delta: np.ndarray, x: np.ndarray, y: np.ndarray):
        if self._args[0].is_variable:
            dx = _unbroadcast_to(delta / y, x.shape)
        else:
            dx = None
        if self._args[1].is_variable:
            dy = _unbroadcast_to(-delta * x / (y ** 2), y.shape)
        else:
            dy = None
        return dx, dy


@_typecheck(exclude_args=('x', 'y'))
def divide(x: Array, y: Array, name: str = None) -> Array:
    """Return element-wise division of two arrays.

    Parameters
    ----------
    x : Array
        Input array.
    y : Array
        Another input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Element-wise division of two arrays.

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.divide([[1, 2], [2, 3]], [-1, 2])
    array([[-1. ,  1. ],
           [-2. ,  1.5]])
    """
    return _Divide(x, y, name=name).forward()
