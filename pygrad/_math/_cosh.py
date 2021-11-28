import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _cosh(x: TensorLike):
    def grad(dout):
        return np.sinh(x) * dout
    return np.cosh(x), grad


def cosh(x: TensorLike) -> Tensor:
    r"""Return hyperbolic cosine of each element.

    .. math::
        \cosh x &= {e^{x} + e^{-x}\over 2}

        {\partial\over\partial x}\cosh x &= \sinh x

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Hyperbolic cosine of each element

    Examples
    --------
    >>> gd.cosh([0, 1, 2])
    Tensor([1.        , 1.54308063, 3.76219569])
    """
    return _cosh(x)
