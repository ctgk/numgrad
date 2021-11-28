import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _normal(
    loc: TensorLike,
    scale: TensorLike,
    *,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
):
    size = size if size is not None else np.broadcast(loc, scale).shape
    n = np.random.normal(size=size).astype(loc.dtype)

    def grad(dout):
        dloc = _unbroadcast_to(dout, loc.shape)
        dscale = _unbroadcast_to(dout * n, scale.shape)
        return dloc, dscale

    return loc + scale * n, grad


def normal(
    loc: TensorLike,
    scale: TensorLike,
    size: tp.Union[int, tp.Tuple[int, ...], tp.List[int], None] = None,
) -> Tensor:
    r"""Return array with normally distributed values.

    .. math::
        \mathcal{N}(x|\mu, \sigma^2) = {1\over\sqrt{2\pi\sigma^2}}
            \exp\left\{-{1\over2\sigma^2}(x-\mu)^2\right\}

    Parameters
    ----------
    loc : TensorLike
        Location parameter of the normal distribution.
    scale : TensorLike
        Scale parameter of the normal distribution.
    size : tp.Union[int, tp.Tuple[int, ...], tp.List[int], None], optional
        Size of the resulting array, by default None

    Returns
    -------
    Tensor
        Tensor with normally distributed values.

    Examples
    --------
    >>> np.random.seed(0)
    >>> gd.random.normal(0, 1, (4,))
    Tensor([1.76405235, 0.40015721, 0.97873798, 2.2408932 ])
    >>> gd.random.normal([-1, 2, 0], 0.1, (3,))
    Tensor([-0.8132442 ,  1.90227221,  0.09500884])
    """
    return _normal(loc, scale, size=size)
