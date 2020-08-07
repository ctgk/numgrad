import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Exp(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        self.output = np.exp(x)
        return self.output

    def _backward_numpy(self, dy, *args):
        return dy * self.output


@_typecheck(exclude=('x',))
def exp(x: Array, *, name: str = None) -> Array:
    """Return exponential of each element

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Exponential of each element

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.exp(1)
    array(2.71828183)
    """
    return _Exp(x, name=name).forward()
