import numpy as np

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args


class _Tanh(_Operator):

    def __init__(self, x: Array, name: str = None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        self.output = np.tanh(x)
        return self.output

    def _backward_numpy(self, dy, x):
        return dy * (1 - np.square(self.output))


@_typecheck_args
def tanh(x, *, name: str = None) -> Array:
    r"""Return hyperbolic tangent of each element.

    .. math::
        \tanh x &= {e^{x} - e^{-x}\over e^{x} + e^{-x}}

        {\partial\over\partial x}\tanh x &= 1 - \tanh^2 x

    Parameters
    ----------
    x
        Input array.
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Hyperbolic tangent of each element

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.tanh([0, 1, 2])
    array([0.        , 0.76159416, 0.96402758])
    """
    return _Tanh(x, name=name).forward()
