import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _transpose(x: TensorLike, *, axes: tp.Optional[tp.Tuple[int, ...]] = None):

    def grad(dout):
        if axes is None:
            return np.transpose(dout)
        return np.transpose(dout, np.argsort(axes))

    return np.transpose(x, axes), grad


def transpose(x, axes: tp.Optional[tp.Tuple[int, ...]] = None) -> Tensor:
    """Return a transposed array.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    axes : tp.Optional[tp.Tuple[int, ...]], optional
        Order of axes of new shape.

    Returns
    -------
    Tensor
        Transposed array.

    Examples
    --------
    >>> gd.transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    Tensor([[1., 4., 7.],
            [2., 5., 8.],
            [3., 6., 9.]])
    >>> gd.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]).transpose(1, 2, 0)
    Tensor([[[1.],
             [2.],
             [3.]],
    <BLANKLINE>
            [[4.],
             [5.],
             [6.]],
    <BLANKLINE>
            [[7.],
             [8.],
             [9.]]])
    """
    return _transpose(x, axes=axes)
