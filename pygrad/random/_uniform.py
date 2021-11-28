import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _uniform(
    low: TensorLike,
    high: TensorLike,
    *,
    size: tp.Union[int, tp.Tuple[int, ...], None] = None,
):
    size = size if size is not None else np.broadcast(low, high).shape
    u = np.random.uniform(0, 1, size=size).astype(low.dtype)

    def grad(dout):
        du = dout * u
        dlow = _unbroadcast_to(dout - du, low.shape)
        dhigh = _unbroadcast_to(du, high.shape)
        return dlow, dhigh

    return low + (high - low) * u, grad


def uniform(
    low: TensorLike,
    high: TensorLike,
    size: tp.Union[int, tp.Tuple[int, ...], None] = None,
) -> Tensor:
    r"""Return array with uniformly distributed values.

    .. math::
        \mathcal{U}(x|a, b) = \begin{cases}
            {1\over b - a} & a \le x \le b\\
            0 & {\rm otherwise}
        \end{cases}

    Parameters
    ----------
    low : TensorLike
        Lower boundary of the output interval.
    high : TensorLike
        Upper boundary of the output interval.
    size : tp.Union[int, tp.Tuple[int, ...], None], optional
        Output shape, by default None

    Returns
    -------
    Tensor
        Tensor with uniformly distributed values.

    Examples
    --------
    >>> np.random.seed(0)
    >>> gd.random.uniform(0, 1, 4)
    Tensor([0.5488135 , 0.71518937, 0.60276338, 0.54488318])
    >>> gd.random.uniform([1, -2], [2, 0], size=2)
    Tensor([ 1.4236548 , -0.70821177])
    """
    return _uniform(low, high, size=size)
