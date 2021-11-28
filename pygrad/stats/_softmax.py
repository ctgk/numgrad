import scipy.special as sp

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _softmax(x: TensorLike, *, axis: int = -1):
    out = sp.softmax(x, axis=axis)

    def grad(dout):
        dx = out * dout
        dx -= out * dx.sum(axis=axis, keepdims=True)
        return dx

    return out, grad


def softmax(x: TensorLike, axis: int = -1) -> Tensor:
    r"""Softmax activation along the given axis.

    .. math::
        {e^{x_i}\over\sum_{n=0}^{N-1}e^{x_n}}

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    axis : int, optional
        Axis to sum along, by default -1

    Returns
    -------
    Tensor
        Result of softmax activation

    Examples
    --------
    >>> gd.stats.softmax([0, 1, -1])
    Tensor([0.24472847, 0.66524096, 0.09003057])
    """
    return _softmax(x, axis=axis)
