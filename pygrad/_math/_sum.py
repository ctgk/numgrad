import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _sum(
    x: TensorLike,
    *,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
):
    out = np.sum(x, axis=axis, keepdims=keepdims)
    if isinstance(axis, int):
        axis = (axis,)

    def grad(dout):
        if all((
            isinstance(dout, np.ndarray),
            (not keepdims),
            (axis is not None),
        )):
            axis_positive = []
            for ax in axis:
                if ax < 0:
                    axis_positive.append(x.ndim + ax)
                else:
                    axis_positive.append(ax)
            for ax in sorted(axis_positive):
                dout = np.expand_dims(dout, ax)
        dx = np.broadcast_to(dout, x.shape)
        return dx

    return out, grad


def sum(
    x: TensorLike,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> Tensor:
    """Sum elements in the array along given axis.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    axis : Union[int, Iterable[int], None], optional
        Axis to sum along, by default None
    keepdims : bool, optional
        Whether to keep dimensionality of the array or not, by default False

    Returns
    -------
    Tensor
        Summation.

    Examples
    --------
    >>> gd.Tensor([1, 2, 3]).sum()
    Tensor(6.)
    >>> gd.sum([[1, 2], [4, 8]], axis=1, keepdims=True)
    Tensor([[ 3.],
            [12.]])
    """
    return _sum(x, axis=axis, keepdims=keepdims)
