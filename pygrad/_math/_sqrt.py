import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Sqrt(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        self.output = np.sqrt(x)
        return self.output

    def _backward_numpy(self, delta, x):
        return 0.5 * delta / self.output


@_typecheck(exclude=('x',))
def sqrt(x: Array, *, name: str = None) -> Array:
    """Return square root of each element

    Parameters
    ----------
    x : Array
        Input array.

    Returns
    -------
    Array
        Square root of each element

    Examples
    --------
    >>> import pygrad as bs
    >>> bs.sqrt([1, 4, 9])
    array([1., 2., 3.])
    """
    return _Sqrt(x, name=name).forward()
