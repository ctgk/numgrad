import numpy as np

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args


class _Exp(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        self.output = np.exp(x)
        return self.output

    def _backward_numpy(self, dy, *args):
        return dy * self.output


@_typecheck_args
def exp(x, *, name: str = None) -> Array:
    """Return exponential of each element

    Parameters
    ----------
    x
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
