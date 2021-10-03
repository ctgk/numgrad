import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Sinh(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        return np.sinh(x)

    def _backward_numpy(self, dy, x):
        return dy * np.cosh(x)


@_typecheck(exclude_args=('x',))
def sinh(x: Array, *, name: str = None) -> Array:
    r"""Return hyperbolic sine of each element.

    .. math::
        \sinh x &= {e^{x} - e^{-x}\over 2}

        {\partial\over\partial x}\sinh x &= \cosh x

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Hyperbolic sine of each element

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.sinh([0, 1, 2])
    array([0.        , 1.17520119, 3.62686041])
    """
    return _Sinh(x, name=name).forward()
