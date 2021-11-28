import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _exp(x: TensorLike):
    out = np.exp(x)

    def grad(dout):
        return out * dout

    return out, grad


def exp(x: TensorLike) -> Tensor:
    """Return exponential of each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Exponential of each element

    Examples
    --------
    >>> gd.exp(1)
    Tensor(2.71828183)
    """
    return _exp(x)
