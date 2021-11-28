import typing as tp

import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.nn._utils import _col2im, _im2col, _to_pair


@_typecheck()
@differentiable_operator
def _max_pool2d(
    x: TensorLike,
    *,
    size: tp.Tuple[int, int] = (2, 2),
    strides: tp.Optional[tp.Tuple[int, int]] = None,
    pad: tp.Tuple[int, int] = (0, 0),
):
    pad = (0,) + pad + (0,)
    x = np.pad(x, tuple((p,) for p in pad), mode='constant')
    padded_shape = x.shape
    col = _im2col(x, size, size if strides is None else strides)
    n, h, w, kh, kw, c = col.shape
    col = col.reshape(n, h, w, kh * kw, c)

    def grad(dout):
        dout_col = np.zeros(dout.shape + (size[0] * size[1],), dtype=x.dtype)
        index = np.where(dout == dout) + (col.argmax(axis=3).ravel(),)
        dout_col[index] = dout.ravel()
        dout_col = np.reshape(dout_col, dout.shape + size)
        dout_col = dout_col.transpose(0, 1, 2, 4, 5, 3)
        dx = np.zeros(padded_shape, dtype=x.dtype)
        _col2im(dout_col, size if strides is None else strides, out=dx)
        slices = tuple(slice(p, l - p) for p, l in zip(pad, padded_shape))
        dx = dx[slices]
        return dx

    return col.max(axis=3), grad


def max_pool2d(
    x: TensorLike,
    size: tp.Union[int, tp.Iterable[int]],
    strides: tp.Union[int, tp.Iterable[int], None] = None,
    pad: tp.Union[int, tp.Iterable[int]] = (0, 0),
):
    """Two-dimensional spatial max pooling.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object (batch_size, height, width, channel).
    size : tp.Union[int, tp.Iterable[int]]
        Pooling size for each axis
    strides : tp.Union[int, tp.Iterable[int], None], optional
        Strides to apply pooling for each axis, by default None
    pad : tp.Union[int, tp.Iterable[int]], optional
        Padding width for each axis, by default (0, 0)
    name : tp.Union[str, None], optional
        The name of the operation, by default None

    Returns
    -------
    Tensor
        Max pooled array

    Examples
    --------
    >>> gd.nn.max_pool2d(gd.Tensor([
    ...     [0, 1, 2, 3],
    ...     [1, 4, -1, 1],
    ...     [8, 0, 2, 5],
    ...     [7, -1, 9, 0]]).reshape(1, 4, 4, 1), 2)
    Tensor([[[[4.],
              [3.]],
    <BLANKLINE>
             [[8.],
              [9.]]]])
    """
    size = _to_pair(size, 'size')
    if strides is not None:
        strides = _to_pair(strides, 'strides')
    pad = _to_pair(pad, 'pad')
    return _max_pool2d(x, size=size, strides=strides, pad=pad)


class MaxPool2D(Module):
    """Two-dimensional spatial pooling layer.

    Examples
    --------
    >>> import pygrad as gd; import numpy as np
    >>> m = gd.nn.MaxPool2D(size=2, strides=2, pad=1)
    >>> m(np.random.rand(2, 5, 5, 4)).shape
    (2, 3, 3, 4)
    """

    @_typecheck()
    def __init__(
        self,
        size: tp.Union[int, tp.Iterable[int]],
        strides: tp.Union[int, tp.Iterable[int], None] = None,
        pad: tp.Union[int, tp.Iterable[int]] = (0, 0),
    ):
        """Initialize max pooling layer.

        Parameters
        ----------
        size : tp.Union[int, tp.Iterable[int]]
            Size of max pooling
        strides : tp.Union[int, tp.Iterable[int], None], optional
            Stride to apply max pooling, by default None
        pad : tp.Union[int, tp.Iterable[int]], optional
            Pad width, by default (0, 0)
        """
        super().__init__()
        self._size = _to_pair(size, 'size')
        self._strides = strides if strides is None else _to_pair(
            strides, 'strides')
        self._pad = _to_pair(pad, 'pad')

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:  # noqa: D102
        return _max_pool2d(
            x, size=self._size, strides=self._strides, pad=self._pad)
