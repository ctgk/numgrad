import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _sin(x: TensorLike):
    def grad(dout):
        return np.cos(x) * dout
    return np.sin(x), grad


def sin(x: TensorLike) -> Tensor:
    """Return trigonometric sine of each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Trigonometric sine of each element

    Examples
    --------
    >>> from math import pi
    >>> gd.sin([0, -pi / 6, 17 * pi / 6, -pi / 2])
    Tensor([ 0. , -0.5,  0.5, -1. ])
    """
    return _sin(x)
