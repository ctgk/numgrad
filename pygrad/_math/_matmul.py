import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _MatMul(_Operator):

    def __init__(self, x, y, name: str = None):
        super().__init__(x, y, name=name)
        if not (self._args[0].ndim == self._args[1].ndim == 2):
            raise ValueError('Arguments of matmul() must be 2-dimensional.')

    @staticmethod
    def _forward_numpy(x, y):
        return x @ y

    @staticmethod
    def _backward_numpy(delta: np.ndarray, x: np.ndarray, y: np.ndarray):
        dx = delta @ y.T
        dy = x.T @ delta
        return dx, dy


@_typecheck(exclude=('x', 'y'))
def matmul(x: Array, y: Array, name: str = None) -> Array:
    """Return matrix multiplication of two arrays.

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
        matrix multiplication of two arrays.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.matmul([[1, 2], [2, 3]], [[-1], [3]])
    array([[5],
           [7]])
    """
    return _MatMul(x, y).forward()
