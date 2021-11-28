import typing as tp

import numpy as np
import scipy.special as sp

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _gumbel_softmax(
    logits: TensorLike,
    *,
    temperature: float,
    axis: int,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
):
    size = logits.shape if size is None else size
    g = np.random.gumbel(size=size).astype(logits.dtype)
    out = sp.softmax(logits + g / temperature, axis=axis)

    def grad(dout):
        dx = out * dout
        dx -= out * dx.sum(axis=axis, keepdims=True)
        dlogits = _unbroadcast_to(dx / temperature, logits.shape)
        return dlogits

    return out, grad


def gumbel_softmax(
    logits: TensorLike,
    temperature: float = 0.1,
    axis: int = -1,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
) -> Tensor:
    """Return random sample from gumbel softmax distribution.

    Parameters
    ----------
    logits : TensorLike
        Logits of probabilities.
    temperature : float, optional
        Temperature parameter for smoothing softmax activation, by default 0.1
    axis : int, optional
        Axis of probabilities, by default -1
    size : tp.Union[int, tp.Tuple[int, ...], tp.List[int], None], optional
        Size of the resulting array, by default None

    Returns
    -------
    Tensor
        Samples from gumbel softmax distribution.

    Examples
    --------
    >>> np.random.seed(0)
    >>> gd.random.gumbel_softmax([2, 0, -1], size=(3, 3))
    Tensor([[9.87461184e-01, 1.39546848e-03, 1.11433480e-02],
            [1.72606944e-01, 8.26853058e-01, 5.39997891e-04],
            [9.99999816e-01, 1.82637560e-07, 1.23878050e-09]])
    """
    return _gumbel_softmax(
        logits, temperature=temperature, axis=axis, size=size)
