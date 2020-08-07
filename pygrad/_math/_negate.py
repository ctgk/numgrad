from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Negate(_Operator):

    def __init__(self, x, name: str = None):
        super().__init__(x, name=name)

    @staticmethod
    def _forward_numpy(x):
        return -x

    @staticmethod
    def _backward_numpy(dy, x):
        return -dy


@_typecheck(exclude=('x',))
def negate(x: Array, *, name: str = None) -> Array:
    """Negate each element

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Negation of each element.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.negate([1, -2, 3])
    array([-1,  2, -3])
    """
    return _Negate(x, name=name).forward()
