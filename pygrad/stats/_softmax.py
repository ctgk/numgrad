import scipy.special as sp

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Softmax(_Operator):

    def __init__(self, x: Array, axis: int = -1, name: str = None):
        super().__init__(x, name=name)
        self._axis = axis

    def _forward_numpy(self, x):
        self.output = sp.softmax(x, axis=self._axis)
        return self.output

    def _backward_numpy(self, delta, x):
        dx = self.output * delta
        dx -= self.output * dx.sum(axis=self._axis, keepdims=True)
        return dx


@_typecheck(exclude_args=('x',))
def softmax(x: Array, axis: int = -1, *, name: str = None) -> Array:
    r"""Softmax activation along the given axis.

    .. math::
        {e^{x_i}\over\sum_{n=0}^{N-1}e^{x_n}}

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
        Result of softmax activation

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.stats.softmax([0, 1, -1])
    array([0.24472847, 0.66524096, 0.09003057])
    """
    return _Softmax(x, axis=axis, name=name).forward()
