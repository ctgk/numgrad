import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad.nn._utils import _im2col, _col2im, _to_pair


class _MaxPool2D(_Operator):

    def __init__(
            self,
            x: Array,
            size: tp.Iterable[int] = (2, 2),
            strides: tp.Union[tp.Iterable[int], None] = None,
            pad: tp.Iterable[int] = (0, 0),
            name: str = None):
        super().__init__(x, name=name)
        self._size = size
        self._strides = strides
        self._pad = (0,) + pad + (0,)

    def _forward_numpy(self, x):
        x = np.pad(x, tuple((p,) for p in self._pad), mode='constant')
        self._padded_shape = x.shape
        col = _im2col(
            x,
            self._size,
            self._size if self._strides is None else self._strides)
        n, h, w, kh, kw, c = col.shape
        col = col.reshape(n, h, w, kh * kw, c)
        self._index = col.argmax(axis=3)
        return col.max(axis=3)

    def _backward_numpy(self, delta, x):
        delta_col = np.zeros(
            delta.shape + (self._size[0] * self._size[1],), dtype=x.dtype)
        index = np.where(delta == delta) + (self._index.ravel(),)
        delta_col[index] = delta.ravel()
        delta_col = np.reshape(delta_col, delta.shape + self._size)
        delta_col = delta_col.transpose(0, 1, 2, 4, 5, 3)
        dx = np.zeros(self._padded_shape, dtype=x.dtype)
        _col2im(
            delta_col,
            self._size if self._strides is None else self._strides,
            out=dx)
        slices = tuple(
            slice(p, len_ - p) for p, len_
            in zip(self._pad, self._padded_shape))
        dx = dx[slices]
        return dx


def max_pool2d(
        x: Array,
        size: tp.Union[int, tp.Iterable[int]],
        strides: tp.Union[int, tp.Iterable[int], None] = None,
        pad: tp.Union[int, tp.Iterable[int]] = (0, 0),
        *,
        name: tp.Union[str, None] = None) -> Array:
    """Two-dimesional spatial max pooling

    Parameters
    ----------
    x : Array
        Input array (batch_size, height, width, channel)
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
    Array
        Max pooled array

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.nn.max_pool2d(pg.Array([
    ...     [0, 1, 2, 3],
    ...     [1, 4, -1, 1],
    ...     [8, 0, 2, 5],
    ...     [7, -1, 9, 0]]).reshape(1, 4, 4, 1), 2)
    array([[[[4],
             [3]],
    <BLANKLINE>
            [[8],
             [9]]]])
    """
    size = _to_pair(size, 'size')
    if strides is not None:
        strides = _to_pair(strides, 'strides')
    pad = _to_pair(pad, 'pad')
    return _MaxPool2D(x, size, strides, pad, name=name).forward()


class MaxPool2D(Module):
    """Two-dimensional spatial pooling layer

    Examples
    --------
    >>> import pygrad as pg; import numpy as np
    >>> m = pg.nn.MaxPool2D(size=2, strides=2, pad=1)
    >>> m(np.random.rand(2, 5, 5, 4)).shape
    (2, 3, 3, 4)
    """

    @_typecheck()
    def __init__(
            self,
            size: tp.Union[int, tp.Iterable[int]],
            strides: tp.Union[int, tp.Iterable[int], None] = None,
            pad: tp.Union[int, tp.Iterable[int]] = (0, 0)):
        super().__init__()
        self._size = _to_pair(size, 'size')
        self._strides = _to_pair(strides, 'strides')
        self._pad = _to_pair(pad, 'pad')

    def __call__(self, x: Array) -> Array:
        return _MaxPool2D(x, self._size, self._strides, self._pad).forward()
