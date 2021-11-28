import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _square(x: TensorLike):
    def grad(dout):
        return 2 * x * dout
    return np.square(x), grad


def square(x: TensorLike) -> Tensor:
    """Return square of each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.



    Returns
    -------
    Tensor
        Square of each element

    Examples
    --------
    >>> gd.square([1, 2, -3])
    Tensor([1., 4., 9.])
    """
    return _square(x)
