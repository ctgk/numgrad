import numpy as np
import scipy.special as sp

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _LogSoftmax(_Operator):

    def __init__(self, x: Array, axis: int = -1, name: str = None):
        super().__init__(x, name=name)
        self._axis = axis

    def _forward_numpy(self, x):
        self.output = sp.log_softmax(x, axis=self._axis)
        return self.output

    def _backward_numpy(self, delta, x):
        dx = delta - np.exp(self.output) * delta.sum(
            axis=self._axis, keepdims=True)
        return dx


@_typecheck(exclude_args=('x',))
def log_softmax(x: Array, axis: int = -1, *, name: str = None) -> Array:
    r"""Return logarithm of softmax activation along the given axis.

    .. math::
        \ln{e^{x_i}\over\sum_{n=0}^{N-1}e^{x_n}}

    Parameters
    ----------
    x : Array
        Input array.
    axis : int, optional
        Axis to sum along, by default -1
    name : str, optional
        Name of the operation, by default None

    Returns
    -------
    Array
        Logarithm of softmax activation

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.stats.log_softmax([0, 1, -1])
    array([-1.40760596, -0.40760596, -2.40760596])
    """
    return _LogSoftmax(x, axis=axis, name=name).forward()
