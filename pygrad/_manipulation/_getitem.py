import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _getitem(
    x: TensorLike,
    *,
    index: tp.Union[int, slice, tp.Tuple[tp.Union[int, slice], ...]],
):
    def grad(dout):
        dx = np.zeros_like(x)
        dx[index] = dout
        return dx
    return x[index], grad


def _getitem_from_tensor(
    x: TensorLike,
    index: tp.Union[int, slice, tp.Tuple[tp.Union[int, slice], ...]],
) -> Tensor:
    return _getitem(x, index=index)
