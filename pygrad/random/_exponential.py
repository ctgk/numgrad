import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _exponential(
    scale: TensorLike,
    *,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
):
    size = scale.shape if size is None else size
    e = np.random.standard_exponential(size).astype(scale.dtype)

    def grad(dout):
        return _unbroadcast_to(e * dout, scale.shape)

    return scale * e, grad


def exponential(
    scale: TensorLike,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
) -> Tensor:
    r"""Return random samples from exponential distribution.

    .. math::
        p(x|\beta) = {1\over\beta}e^{-{x\over\beta}}

    Parameters
    ----------
    scale : Array
        Scale parameter of exponential distribution. Larger this value is,
        larger the resulting samples.
    size : tp.Union[int, tp.Tuple[int, ...], tp.List[int], None], optional
        Size of returned random samples, by default None

    Returns
    -------
    Tensor
        Random samples from the distribution
    """
    return _exponential(scale, size=size)
