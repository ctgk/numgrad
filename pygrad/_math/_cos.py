import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _cos(x: TensorLike):

    def grad(dout):
        return -np.sin(x) * dout

    out = np.cos(x)
    return out, grad


def cos(x: TensorLike) -> Tensor:
    """Return trigonometric cosine of each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Trigonometric cosine of each element

    Examples
    --------
    >>> from math import pi
    >>> gd.cos([0, pi / 3, 14 * pi / 3])
    Tensor([ 1. ,  0.5, -0.5])
    """
    return _cos(x)
