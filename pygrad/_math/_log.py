import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Log(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    @staticmethod
    def _forward_numpy(x):
        return np.log(x)

    @staticmethod
    def _backward_numpy(dy, x):
        return dy / x


@_typecheck(exclude=('x',))
def log(x: Array, *, name: str = None) -> Array:
    """Return natural logarithm of each element

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Natural logarithm of each element

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.log(1)
    array(0.)
    """
    return _Log(x, name=name).forward()
