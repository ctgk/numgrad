import typing as tp

import numpy as np

from pygrad._core._config import config
from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._dtypes import DataType
from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.nn._utils import _col2im, _im2col, _to_pair


@_typecheck()
@differentiable_operator
def _conv2d(
    x: TensorLike,
    w_flat: TensorLike,
    *,
    size: tp.Tuple[int, int],
    strides: tp.Tuple[int, int] = (1, 1),
    pad: tp.Tuple[int, int] = (0, 0),
):
    pad = (0,) + pad + (0,)
    x = np.pad(x, tuple((p,) for p in pad), mode='constant')
    col = _im2col(x, size, strides)
    out_shape = col.shape[:3] + (w_flat.shape[1],)
    col_flat = col.reshape(-1, w_flat.shape[0])

    def grad(dout):
        dout_flat = dout.reshape(-1, dout.shape[-1])
        dcol = dout_flat @ w_flat.T
        dcol = dcol.reshape(col.shape)
        dx = np.zeros(x.shape, dtype=x.dtype)
        _col2im(dcol, strides, out=dx)
        slices = tuple(slice(p, s - p) for p, s in zip(pad, x.shape))
        dx = dx[slices]
        dw_flat = col_flat.T @ dout_flat
        return dx, dw_flat

    return np.matmul(col_flat, w_flat).reshape(out_shape), grad


def conv2d(
    x: TensorLike,
    w: TensorLike,
    strides: tp.Union[int, tp.Tuple[int, int]] = (1, 1),
    pad: tp.Union[int, tp.Tuple[int, int]] = (0, 0),
) -> Tensor:
    """Perform 2d-convolution operation.

    This actually computes correlation.

    Parameters
    ----------
    x : TensorLike
        Input array (batch_size, height, width, in_channels)
    w : TensorLike
        Kernel (kh, kw, in_channels, out_channels)
    strides : tp.Union[int, tp.Tuple[int, int]], optional
        Stride to apply convolution, by default (1, 1)
    pad : tp.Union[int, tp.Tuple[int, int]], optional
        Pad width before convolution, by default (0, 0)

    Returns
    -------
    Tensor
        Convolution of two arrays

    Examples
    --------
    >>> gd.nn.conv2d(
    ...     np.arange(16).reshape(1, 4, 4, 1),
    ...     np.arange(9).reshape(3, 3, 1, 1))
    Tensor([[[[258.],
              [294.]],
    <BLANKLINE>
             [[402.],
              [438.]]]])
    """
    strides = _to_pair(strides, 'strides')
    pad = _to_pair(pad, 'pad')
    return _conv2d(
        x,
        w.reshape(-1, w.shape[-1]),
        size=w.shape[:2],
        strides=strides,
        pad=pad,
    )


class Conv2D(Module):
    """Two-dimensional convolution layer.

    Examples
    --------
    >>> c = gd.nn.Conv2D(1, 10, 3, strides=1, pad=0, bias=True)
    >>> c(np.random.rand(2, 5, 5, 1)).shape
    (2, 3, 3, 10)
    """

    @_typecheck()
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tp.Union[int, tp.Iterable[int]],
        strides: tp.Union[int, tp.Iterable[int]] = (1, 1),
        pad: tp.Union[int, tp.Iterable[int]] = (0, 0),
        bias: bool = True,
        dtype: tp.Union[tp.Type[DataType], None] = None,
    ):
        """Initialize 2d convolution module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : tp.Union[int, tp.Iterable[int]]
            Size of convolution kernel
        strides : tp.Union[int, tp.Iterable[int]], optional
            Strides of kernel convolution, by default (1, 1)
        pad : tp.Union[int, tp.Iterable[int]], optional
            Pad width, by default (0, 0)
        bias : bool, optional
            Add bias if true, by default True
        dtype : tp.Union[tp.Type[DataType], None], optional
            Data type, by default None
        """
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
        self.weight = Tensor(
            np.random.uniform(
                -v, v,
                (kernel_size[0] * kernel_size[1] * in_channels, out_channels)),
            dtype=dtype, is_variable=True)
        if bias:
            self.bias = Tensor(
                np.random.uniform(-v, v, out_channels),
                dtype=dtype,
                is_variable=True)

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:  # noqa: D102
        x = _conv2d(
            x,
            self.weight,  # this if w_flat
            size=self._kernel_size,
            strides=self._strides,
            pad=self._pad,
        )
        if hasattr(self, 'bias'):
            x = x + self.bias
        return x
