import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _tan(x: TensorLike):
    out = np.tan(x)

    def grad(dout):
        return (1 + np.square(out)) * dout

    return out, grad


def tan(x: TensorLike) -> Tensor:
    """Return trigonometric tangent of each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Trigonometric tangent of each element

    Examples
    --------
    >>> from math import pi
    >>> gd.tan([0, pi / 4, -9 * pi / 4])
    Tensor([ 0.,  1., -1.])
    """
    return _tan(x)
