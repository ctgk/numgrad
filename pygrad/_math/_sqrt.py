import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _sqrt(x: TensorLike):
    out = np.sqrt(x)

    def grad(dout):
        return 0.5 / out * dout

    return out, grad


def sqrt(x: TensorLike) -> Tensor:
    """Return square root of each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Square root of each element

    Examples
    --------
    >>> gd.sqrt([1, 4, 9])
    Tensor([1., 2., 3.])
    """
    return _sqrt(x)
