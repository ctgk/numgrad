import numpy as np

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _ReLU(_Operator):

    def __init__(self, x, name=None):
        super().__init__(x, name=name)

    @staticmethod
    def _forward_numpy(x):
        return np.maximum(x, 0)

    @staticmethod
    def _backward_numpy(delta, x):
        return delta * (x > 0)


@_typecheck(exclude=('x',))
def relu(x: Array, *, name: str = None) -> Array:
    r"""Element-wise rectified linear unit.

    .. math::
        {\rm ReLU}(x) = \max(x, 0)

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        The name of the operation, by default None

    Returns
    -------
    Array
        The output of rectified linear unit.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.nn.relu([1, -1, 2, -3])
    array([1., 0., 2., 0.])
    """
    return _ReLU(x, name=name).forward()


class ReLU(Module):
    """Rectified linear unit layer
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x: Array, **kwargs) -> Array:
        return _ReLU(x).forward()
