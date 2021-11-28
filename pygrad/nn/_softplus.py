import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _softplus(x: TensorLike):

    def grad(dout):
        return (np.tanh(0.5 * x) * 0.5 + 0.5) * dout

    out = np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))
    return out, grad


def softplus(x: TensorLike) -> Tensor:
    """Return element-wise softplus activation of the input.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object

    Returns
    -------
    Tensor
        Element-wise softplus activation of the input.
    """
    return _softplus(x)
