import typing as tp

import numpy as np
import scipy.special as sp

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Logsumexp(_Operator):

    def __init__(
            self,
            x: Array,
            axis: tp.Union[int, tp.Iterable[int], None] = None,
            keepdims: bool = False,
            name: str = None):
        super().__init__(x, name=name)
        self._axis = (axis,) if isinstance(axis, int) else axis
        self._keepdims = keepdims

    def _forward_numpy(self, x):
        self.output = sp.logsumexp(x, axis=self._axis, keepdims=self._keepdims)
        return self.output

    def _backward_numpy(self, delta, x):
        if all((
            isinstance(delta, np.ndarray),
            (not self._keepdims),
            (self._axis is not None),
        )):
            axis_positive = []
            for axis in self._axis:
                if axis < 0:
                    axis_positive.append(x.ndim + axis)
                else:
                    axis_positive.append(axis)
            for axis in sorted(axis_positive):
                delta = np.expand_dims(delta, axis)
                self.output = np.expand_dims(self.output, axis)
        delta = np.broadcast_to(delta, x.shape)
        self.output = np.broadcast_to(self.output, x.shape)
        return delta * np.exp(x - self.output)


@_typecheck(exclude_args=('x',))
def logsumexp(x: Array,
              axis: tp.Union[int, tp.Iterable[int], None] = None,
              keepdims: bool = False,
              *,
              name: str = None) -> Array:
    r"""Return natural logarithm of summation of exponentials along give axis.

    .. math::
        f({\boldsymbol x}) = \ln\Sigma_{i=0}^{N-1}e^{x_i}

    Parameters
    ----------
    x : Array
        Input array.
    axis : tp.Union[int, tp.Iterable[int], None], optional
        Axis to sum along, by default None
    keepdims : bool, optional
        Whether to keep dimensionality of the array or not, by default False
    name : str, optional
        Name of the operation, by default None

    Returns
    -------
    Array
        Natural logarithm of summation of exponentials

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.logsumexp([0, 1, -1])
    array(1.40760596)
    """
    return _Logsumexp(x, axis=axis, keepdims=keepdims, name=name).forward()
