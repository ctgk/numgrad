import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _mean(
    x: TensorLike,
    *,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
):
    out = np.mean(x, axis=axis, keepdims=keepdims)

    def grad(dout):
        if all((
            isinstance(dout, np.ndarray),
            (not keepdims),
            (axis is not None),
        )):
            axis_positive = []
            for ax in axis if isinstance(axis, tuple) else (axis,):
                if ax < 0:
                    axis_positive.append(x.ndim + ax)
                else:
                    axis_positive.append(ax)
            for ax in sorted(axis_positive):
                dout = np.expand_dims(dout, ax)
        dx = np.broadcast_to(dout, x.shape)
        dx = dx * dout.size / x.size
        return dx

    return out, grad


def mean(
    x: TensorLike,
    axis: tp.Union[int, tp.Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> Tensor:
    """Mean of elements in the array along given axis.

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
        Mean of the input array.

    Examples
    --------
    >>> gd.Tensor([1, 2, 3]).mean()
    Tensor(2.)
    >>> gd.mean([[1., 2.], [4., 8.]], axis=1, keepdims=True)
    Tensor([[1.5],
            [6. ]])
    """
    return _mean(x, axis=axis, keepdims=keepdims)
