import numpy as np
import scipy.special as sp

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _log_softmax(x: TensorLike, *, axis: int = -1):
    out = sp.log_softmax(x, axis=axis)

    def grad(dout):
        return dout - np.exp(out) * dout.sum(axis=axis, keepdims=True)

    return out, grad


def log_softmax(x: TensorLike, axis: int = -1) -> Tensor:
    r"""Return logarithm of softmax activation along the given axis.

    .. math::
        \ln{e^{x_i}\over\sum_{n=0}^{N-1}e^{x_n}}

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    axis : int, optional
        Axis to sum along, by default -1

    Returns
    -------
    Tensor
        Logarithm of softmax activation

    Examples
    --------
    >>> gd.stats.log_softmax([0, 1, -1])
    Tensor([-1.40760596, -0.40760596, -2.40760596])
    """
    return _log_softmax(x, axis=axis)
