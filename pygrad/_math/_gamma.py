import scipy.special as sp

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _gamma(x: TensorLike):
    out = sp.gamma(x)

    def grad(dout):
        return sp.digamma(x) * out * dout
    return out, grad


def gamma(x: TensorLike) -> Tensor:
    r"""Element-wise gamma function.

    .. math::
        \Gamma(z) &= \int^{\infty}_0 e^{-t}t^{z-1}{\rm d}t

        {{\rm d}\over{\rm d} z}\Gamma(z) &= \psi(z)\Gamma(z),

    where :math:`\psi(z)={{\rm d}\over{\rm d} z}\ln\Gamma(z)`.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Value of element-wise gamma function.

    Examples
    --------
    >>> gd.gamma([1, 2, 3, 4])
    Tensor([1., 1., 2., 6.])
    """
    return _gamma(x)
