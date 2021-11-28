import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _dropout(x: TensorLike, *, droprate: float = None):
    mask = np.random.choice(
        [1 / (1 - droprate), 0], size=x.shape, p=[1 - droprate, droprate])
    if x.dtype != mask.dtype:
        mask = mask.astype(x.dtype)

    def grad(dout):
        return mask * dout

    return x * mask, grad


def dropout(
    x: TensorLike,
    droprate: tp.Union[float, None] = 0.5,
) -> Tensor:
    """Element-wise dropout function.

    Parameters
    ----------
    x : Array
        Input array
    droprate : tp.Union[float, None], optional
        Probability of dropping values, by default 0.5

    Returns
    -------
    Tensor
        The output of dropout function

    Examples
    --------
    >>> np.random.seed(1111)
    >>> gd.nn.dropout([1, -2, 3, -4])
    Tensor([ 2., -0.,  6., -8.])
    """
    if droprate is None:
        return x
    return _dropout(x, droprate=droprate)


class Dropout(Module):
    """Dropout module."""

    @_typecheck()
    def __init__(self, droprate: float = 0.5):
        """Initialize dropout module.

        Parameters
        ----------
        droprate : float, optional
            Rate of dropping values, by default 0.5
        """
        super().__init__()
        self._droprate = droprate

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:  # noqa: D102
        return dropout(
            x,
            droprate=kwargs.get('droprate', self._droprate),
        )
