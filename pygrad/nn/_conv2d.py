import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._module import Module
from pygrad._core._operator import _Operator
from pygrad._core._types import DataType
from pygrad._utils._typecheck import _typecheck
from pygrad.nn._utils import _im2col, _col2im, _to_pair


class _Conv2d(_Operator):

    def __init__(
            self,
            x: Array,
            w: Array,
            size: tp.Iterable[int],
            strides: tp.Iterable[int] = (1, 1),
            pad: tp.Iterable[int] = ((0,), (0,)),
            name: str = None):
        super().__init__(x, w, name=name)
        self._size = size
        self._strides = strides
        self._pad = ((0,),) + pad + ((0,),)

    def _forward_numpy(self, x, w):
        x = np.pad(x, pad_width=self._pad, mode='constant')
        self._padded_shape = x.shape
        col = _im2col(x, self._size, self._strides)
        self._col_shape = col.shape
        self._out_shape = col.shape[:3] + (w.shape[1],)
        self._col_flat = col.reshape(-1, w.shape[0])
        return np.matmul(self._col_flat, w).reshape(self._out_shape)

    def _backward_numpy(self, delta, x, w):
        delta = delta.reshape(-1, delta.shape[-1])
        dcol = delta @ w.T
        dcol = dcol.reshape(self._col_shape)
        dx = np.zeros(self._padded_shape, dtype=x.dtype)
        _col2im(dcol, self._strides, out=dx)
        slices = tuple(
            slice(p[0], len_ - p[0]) for p, len_
            in zip(self._pad, self._padded_shape))
        dx = dx[slices]
        dw = self._col_flat.T @ delta
        return dx, dw


@_typecheck(exclude=('x', 'w'))
def conv2d(
        x: Array,
        w: Array,
        strides: tp.Union[int, tp.Iterable[int]] = (1, 1),
        pad: tp.Union[int, tp.Iterable[int]] = (0, 0),
        *,
        name: tp.Union[str, None] = None) -> Array:
    """Perform convolution operation for neural network

    This actually computes correlaion.

    Parameters
    ----------
    x : Array
        Input array (batch_size, height, width, in_channels)
    w : Array
        Kernel (kh, kw, in_channels, out_channels)
    strides : tp.Union[int, tp.Iterable[int]], optional
        Stride to apply convolution, by default (1, 1)
    pad : tp.Union[int, tp.Iterable[int]], optional
        Pad width before convolution, by default (0, 0)
    name : tp.Union[str, None], optional
        The name of the operation, by default None

    Returns
    -------
    Array
        Convolution of two arrays

    Examples
    --------
    >>> import pygrad as pg; import numpy as np
    >>> pg.nn.conv2d(
    ...     np.arange(16).reshape(1, 4, 4, 1),
    ...     np.arange(9).reshape(3, 3, 1, 1))
    array([[[[258.],
             [294.]],
    <BLANKLINE>
            [[402.],
             [438.]]]])
    """
    strides = _to_pair(strides, 'strides')
    pad = _to_pair(pad, 'pad')
    return _Conv2d(
        x,
        w.reshape(-1, w.shape[-1]),
        size=w.shape[:2],
        strides=strides,
        pad=tuple((p,) for p in pad),
        name=name).forward()


class Conv2D(Module):
    """Two-dimensional convolution layer

    Examples
    --------
    >>> import pygrad as pg; import numpy as np
    >>> c = pg.nn.Conv2D(1, 10, 3, strides=1, pad=0, bias=True)
    >>> c(np.random.rand(2, 5, 5, 1)).shape
    (2, 3, 3, 10)
    """

    @_typecheck()
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tp.Union[int, tp.Iterable[int]],
            strides: tp.Union[int, tp.Iterable[int]],
            pad: tp.Union[int, tp.Iterable[int]],
            bias: bool = True,
            dtype: tp.Union[tp.Type[DataType], None] = None):
        super().__init__()
        dtype = dtype if dtype is not None else config.dtype
        self._in_channels = in_channels
        self._out_channels = out_channels
        kernel_size = _to_pair(kernel_size, 'kernel_size')
        strides = _to_pair(strides, 'strides')
        pad = _to_pair(pad, 'pad')
        v = 1 / np.sqrt(in_channels * kernel_size[0] * kernel_size[1])
        self._kernel_size = kernel_size
        self._strides = strides
        self._pad = pad
        self.weight = Array(
            np.random.uniform(
                -v, v,
                (kernel_size[0] * kernel_size[1] * in_channels, out_channels)),
            dtype=dtype, is_differentiable=True)
        if bias:
            self.bias = Array(
                np.random.uniform(-v, v, out_channels),
                dtype=dtype,
                is_differentiable=True)

    def __call__(self, x: Array) -> Array:
        x = _Conv2d(
            x,
            self.weight,
            size=self._kernel_size,
            strides=self._strides,
            pad=tuple((p,) for p in self._pad)).forward()
        if hasattr(self, 'bias'):
            x = x + self.bias
        return x
