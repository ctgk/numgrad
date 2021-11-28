import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _gumbel_sigmoid(
    logits: TensorLike,
    *,
    temperature: float,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
):
    size = logits.shape if size is None else size
    dg = np.random.gumbel(size=size) - np.random.gumbel(size=size)
    dg = dg.astype(logits.dtype)
    a = (logits + dg) / temperature
    out = np.tanh(0.5 * a) * 0.5 + 0.5

    def grad(dout):
        da = dout * out * (1 - out)
        dlogits = _unbroadcast_to(da / temperature, logits.shape)
        return dlogits

    return out, grad


def gumbel_sigmoid(
    logits: TensorLike,
    temperature: float = 0.1,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
) -> Tensor:
    r"""Return random samples from gumbel sigmoid distributions.

    .. math::
        y = \sigma({x + g\over\tau}),

    Parameters
    ----------
    logits : Tensor
        Logits of probabilities
    temperature : float, optional
        Smoothing parameter of sigmoid activations, by default 0.1
    size : tp.Union[int, tp.Tuple[int, ...], tp.List[int], None], optional
        Size of the resulting array, by default None

    Returns
    -------
    Tensor
        Samples from gumbel sigmoid distributions

    Examples
    --------
    >>> np.random.seed(0)
    >>> gd.random.gumbel_sigmoid([2, -1, 0], size=(1, 3))
    Tensor([[9.99999998e-01, 1.20031814e-08, 7.63747391e-01]])
    """
    return _gumbel_sigmoid(logits, temperature=temperature, size=size)
