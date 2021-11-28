import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _tanh(x: TensorLike):
    out = np.tanh(x)

    def grad(dout):
        return (1 - np.square(out)) * dout

    return out, grad


def tanh(x: TensorLike) -> Tensor:
    r"""Return hyperbolic tangent of each element.

    .. math::
        \tanh x &= {e^{x} - e^{-x}\over e^{x} + e^{-x}}

        {\partial\over\partial x}\tanh x &= 1 - \tanh^2 x

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.



    Returns
    -------
    Tensor
        Hyperbolic tangent of each element

    Examples
    --------
    >>> gd.tanh([0, 1, 2])
    Tensor([0.        , 0.76159416, 0.96402758])
    """
    return _tanh(x)
