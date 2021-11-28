import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _max(
    x: TensorLike,
    *,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
):
    out = np.max(x, axis=axis, keepdims=keepdims)

    def grad(dout):
        if x.ndim == 0:
            return dout
        if all((
            isinstance(dout, np.ndarray),
            (not keepdims),
            (axis is not None),
        )):
            axis_positive = []
            for ax in axis if isinstance(axis, tuple) else (axis,):
                axis_positive.append(x.ndim + ax if ax < 0 else ax)
            for ax in sorted(axis_positive):
                dout = np.expand_dims(dout, ax)
        dx = 1 * np.broadcast_to(dout, x.shape)
        dx[np.where(x != x.max(axis=axis, keepdims=True))] = 0
        return dx

    return out, grad


def max(
    x: TensorLike,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> Tensor:
    """Return maximum element along given axis.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    axis : tp.Union[int, tp.Iterable[int]], optional
        Axis to find maximum value along, by default None
    keepdims : bool, optional
        Whether to keep dimensionality of the array or not, by default False

    Returns
    -------
    Tensor
        Maximum element.

    Examples
    --------
    >>> a = gd.Tensor([[2, 3], [-1, -9]])
    >>> a.max()
    Tensor(3.)
    >>> a.max(axis=0)
    Tensor([2., 3.])
    >>> a.max(axis=1, keepdims=True)
    Tensor([[ 3.],
            [-1.]])
    """
    return _max(x, axis=axis, keepdims=keepdims)
