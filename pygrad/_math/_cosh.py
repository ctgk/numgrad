import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Cosh(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        return np.cosh(x)

    def _backward_numpy(self, dy, x):
        return dy * np.sinh(x)


@_typecheck(exclude=('x',))
def cosh(x: Array, *, name: str = None) -> Array:
    r"""Return hyperbolic cosine of each element.

    .. math::
        \cosh x &= {e^{x} + e^{-x}\over 2}

        {\partial\over\partial x}\cosh x &= \sinh x

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Hyperbolic cosine of each element

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.cosh([0, 1, 2])
    array([1.        , 1.54308063, 3.76219569])
    """
    return _Cosh(x, name=name).forward()
