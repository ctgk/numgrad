import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _sinh(x: TensorLike):
    def grad(dout):
        return np.cosh(x) * dout
    return np.sinh(x), grad


def sinh(x: TensorLike) -> Tensor:
    r"""Return hyperbolic sine of each element.

    .. math::
        \sinh x &= {e^{x} - e^{-x}\over 2}

        {\partial\over\partial x}\sinh x &= \cosh x

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Hyperbolic sine of each element

    Examples
    --------
    >>> gd.sinh([0, 1, 2])
    Tensor([0.        , 1.17520119, 3.62686041])
    """
    return _sinh(x)
