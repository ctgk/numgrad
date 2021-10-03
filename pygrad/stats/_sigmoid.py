import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Sigmoid(_Operator):

    def __init__(self, x, name=None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        self.output = np.tanh(x * 0.5) * 0.5 + 0.5
        return self.output

    def _backward_numpy(self, delta, x):
        return delta * self.output * (1 - self.output)


@_typecheck(exclude_args=('x',))
def sigmoid(x: Array, *, name: str = None) -> Array:
    r"""Element-wise sigmoid function

    .. math::
        \sigma(x) = {1\over1 + e^{-x}}

    Parameters
    ----------
    x : Array
        Input array.
    name : str, optional
        Name of the operation, by default None

    Returns
    -------
    Array
        Output of sigmoid function.

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.stats.sigmoid([-1, 0, 1])
    array([0.26894142, 0.5       , 0.73105858])
    """
    return _Sigmoid(x, name=name).forward()
