import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _sigmoid(x: TensorLike):
    out = np.tanh(x * 0.5) * 0.5 + 0.5

    def grad(dout):
        return out * (1 - out) * dout

    return out, grad


def sigmoid(x: TensorLike) -> Tensor:
    r"""Element-wise sigmoid function.

    .. math::
        \sigma(x) = {1\over1 + e^{-x}}

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Output of sigmoid function.

    Examples
    --------
    >>> gd.stats.sigmoid([-1, 0, 1])
    Tensor([0.26894142, 0.5       , 0.73105858])
    """
    return _sigmoid(x)
