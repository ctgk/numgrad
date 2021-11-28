import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _log(x: TensorLike):
    def grad(dout):
        return dout / x
    return np.log(x), grad


def log(x: TensorLike) -> Tensor:
    """Return natural logarithm of each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Natural logarithm of each element

    Examples
    --------
    >>> gd.log(1)
    Tensor(0.)
    """
    return _log(x)
