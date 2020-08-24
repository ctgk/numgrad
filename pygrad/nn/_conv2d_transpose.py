import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._module import Module
from pygrad._core._operator import _Operator
from pygrad._core._types import DataType
from pygrad._utils._typecheck import _typecheck
from pygrad.nn._utils import _im2col, _col2im, _to_pair


class _Conv2dTranspose(_Operator):

    def __init__(
            self,
            x,
            w,
            size: tp.Iterable[int],
            out_channels: int,
            strides: tp.Iterable[int],
            pad: tp.Iterable[int] = (0, 0),
            shape: tp.Union[tp.Iterable[int], None] = None,
            name: str = None):
        super().__init__(x, w, name=name)
        self._size = size
        self._out_channels = out_channels
        self._strides = strides
        self._pad = pad
        self._shape = shape

    def _forward_numpy(self, x, w):
        shape = tuple(
            s * (len_ - 1) + k for s, len_, k
            in zip(self._strides, x.shape[1:], self._size)
        ) if self._shape is None else self._shape
        shape = tuple(s + 2 * p for s, p in zip(shape, self._pad))
        shape = (len(x),) + shape + (self._out_channels,)
        col_flat = np.matmul(x, w.T)  # (N, Hx, Wx, Hk * Wk * out_ch)
        out = np.zeros(shape, dtype=x.dtype)
        _col2im(
            col_flat.reshape(*col_flat.shape[:3], *self._size, -1),
            self._strides,
            out=out)
        return out[
            :,
            self._pad[0]: out.shape[1] - self._pad[0],
            self._pad[1]: out.shape[2] - self._pad[1]]

    def _backward_numpy(self, delta, x, w):
        delta = np.pad(
            delta,
            ((0,),) + tuple((p,) for p in self._pad) + ((0,),),
            mode='constant')
        dcol = _im2col(delta, self._size, self._strides)

        # (N * Hx * Wx, Hk * Wk * out_ch)
        dcol_flat = dcol.reshape(-1, w.shape[0])

        dx = np.matmul(dcol_flat, w).reshape(x.shape)
        dy = np.matmul(x.reshape(-1, x.shape[-1]).T, dcol_flat).T
        return dx, dy


@_typecheck(exclude_args=('x', 'w'))
def conv2d_transpose(
        x: Array,
        w: Array,
        strides: tp.Union[int, tp.Iterable[int]] = (1, 1),
        pad: tp.Union[int, tp.Iterable[int]] = (0, 0),
        shape: tp.Union[tp.Iterable[int], None] = None,
        *,
        name: tp.Union[str, None] = None) -> Array:
    """Perform transposed convolution operation

    Parameters
    ----------
    x : Array
        Input array (batch_size, height, width, in_channels)
    w : Array
        Kernel (kh, kw, out_channels, in_channels)
    strides : tp.Union[int, tp.Iterable[int]], optional
        Stride to apply convolution for each axis, by default (1, 1)
    pad : tp.Union[int, tp.Iterable[int]], optional
        Pad width for each axis, by default (0, 0)
    shape : tp.Union[tp.Iterable[int], None], optional
        Output shape, by default None
    name : tp.Union[str, None], optional
        The anme of the operation, by default None

    Returns
    -------
    Array
        Transposed convolution of two arrays

    Examples
    --------
    >>> import pygrad as gd; import numpy as np
    >>> gd.nn.conv2d_transpose(
    ...     np.arange(16).reshape(1, 4, 4, 1),
    ...     np.arange(9).reshape(3, 3, 1, 1),
    ...     strides=1,
    ...     pad=1,
    ...     shape=(4, 4))
    array([[[[  7.],
             [ 23.],
             [ 38.],
             [ 41.]],
    <BLANKLINE>
            [[ 45.],
             [102.],
             [138.],
             [126.]],
    <BLANKLINE>
            [[129.],
             [246.],
             [282.],
             [234.]],
    <BLANKLINE>
            [[197.],
             [341.],
             [374.],
             [287.]]]])
    """
    strides = _to_pair(strides, 'strides')
    pad = _to_pair(pad, 'pad')
    return _Conv2dTranspose(
        x, w.reshape(-1, w.shape[-1]), w.shape[:2], w.shape[2],
        strides, pad, shape, name).forward()


class Conv2DTranspose(Module):
    """Two-dimesional tranposed convolution layer

    Examples
    --------
    >>> import pygrad as gd; import numpy as np
    >>> c = gd.nn.Conv2DTranspose(1, 10, 3, strides=1, pad=0, bias=True)
    >>> c(np.random.rand(2, 5, 5, 1)).shape
    (2, 7, 7, 10)
    """

    @_typecheck()
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: tp.Union[int, tp.Iterable[int]],
            strides: tp.Union[int, tp.Iterable[int]],
            pad: tp.Union[int, tp.Iterable[int]] = (0, 0),
            shape: tp.Union[tp.Iterable[int], None] = None,
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
        self._shape = shape
        self.weight = Array(
            np.random.uniform(
                -v, v,
                (kernel_size[0] * kernel_size[1] * out_channels, in_channels)),
            dtype=dtype,
            is_variable=True)
        if bias:
            self.bias = Array(
                np.random.uniform(-v, v, out_channels),
                dtype=dtype,
                is_variable=True)

    def __call__(self, x: Array, **kwargs) -> Array:
        x = _Conv2dTranspose(
            x, self.weight, size=self._kernel_size,
            out_channels=self._out_channels, strides=self._strides,
            pad=self._pad, shape=self._shape).forward()
        if hasattr(self, 'bias'):
            x = x + self.bias
        return x
