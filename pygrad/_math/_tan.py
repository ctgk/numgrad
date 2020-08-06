import numpy as np

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args


class _Tan(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        self.output = np.tan(x)
        return self.output

    def _backward_numpy(self, dy, x):
        return dy * (1 + np.square(self.output))


@_typecheck_args
def tan(x, *, name: str = None) -> Array:
    """Return trigonometric tangent of each element.

    Parameters
    ----------
    x
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Trigonometric tangent of each element

    Examples
    --------
    >>> import pygrad as pg; from math import pi
    >>> pg.tan([0, pi / 4, -9 * pi / 4])
    array([ 0.,  1., -1.])
    """
    return _Tan(x, name=name).forward()
