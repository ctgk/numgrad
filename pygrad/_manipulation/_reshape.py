import typing as tp

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _reshape(x: TensorLike, *, newshape: tp.Tuple[int, ...]):

    def grad(dout):
        return dout.reshape(*x.shape)

    return x.reshape(*newshape), grad


def reshape(x, newshape: tp.Tuple[int, ...]) -> Tensor:
    """Return a reshaped array.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    newshape : tp.Tuple[int, ...]
        New shape

    Returns
    -------
    Tensor
        Reshaped array.

    Examples
    --------
    >>> gd.reshape([1, 2, 3, 4, 5, 6], (2, -1))
    Tensor([[1., 2., 3.],
            [4., 5., 6.]])
    >>> gd.Tensor([1, 2, 3, 4, 5, 6]).reshape(1, 6)
    Tensor([[1., 2., 3., 4., 5., 6.]])
    """
    return _reshape(x, newshape=newshape)
