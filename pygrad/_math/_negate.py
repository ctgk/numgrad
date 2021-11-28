from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _negate(x: TensorLike):
    def grad(dout):
        return -dout
    return -x, grad


def negate(x: TensorLike) -> Tensor:
    """Negate each element.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        Negation of each element.

    Examples
    --------
    >>> gd.negate([1, -2, 3])
    Tensor([-1.,  2., -3.])
    """
    return _negate(x)
