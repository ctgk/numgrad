import typing as tp

import numpy as np

from pygrad._core._config import config
from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._dtypes import DataType
from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.nn._utils import _col2im, _im2col, _to_pair


def _conv2dtr_default_out_shape(input_size, kernel_size, strides):
    return tuple(
        s * (i - 1) + k for i, k, s in zip(input_size, kernel_size, strides)
    )


@_typecheck()
@differentiable_operator
def _conv2d_transpose(
    x: TensorLike,
    w_flat: TensorLike,  # (size[0] * size[1] * out_ch, in_ch)
    *,
    size: tp.Tuple[int, int],
    strides: tp.Tuple[int, int] = (1, 1),
    pad: tp.Tuple[int, int] = (0, 0),
    shape: tp.Optional[tp.Tuple[int, int]] = None,
):
    out_channels = w_flat.shape[0] // (size[0] * size[1])
    shape = (
        _conv2dtr_default_out_shape(x.shape[1:], size, strides)
        if shape is None else shape
    )
    shape = tuple(s + 2 * p for s, p in zip(shape, pad))
    shape = (len(x),) + shape + (out_channels,)
    col_flat = np.matmul(x, w_flat.T)
    out = np.zeros(shape, dtype=x.dtype)
    _col2im(col_flat.reshape(*col_flat.shape[:3], *size, -1), strides, out=out)

    def grad(dout):
        dout_padded = np.pad(
            dout,
            ((0,),) + tuple((p,) for p in pad) + ((0,),),
            mode='constant',
        )
        dcol = _im2col(dout_padded, size, strides)
        dcol_flat = dcol.reshape(-1, w_flat.shape[0])
        dx = np.matmul(dcol_flat, w_flat).reshape(x.shape)
        dw_flat = np.matmul(x.reshape(-1, x.shape[-1]).T, dcol_flat).T
        return dx, dw_flat

    return out[:, pad[0]: shape[1] - pad[0], pad[1]: shape[2] - pad[1]], grad


def conv2d_transpose(
    x: TensorLike,
    w: TensorLike,
    strides: tp.Union[int, tp.List[int], tp.Tuple[int, int]] = (1, 1),
    pad: tp.Union[int, tp.List[int], tp.Tuple[int, int]] = (0, 0),
    shape: tp.Union[tp.List[int], tp.Tuple[int, int], None] = None,
) -> Tensor:
    """Perform transposed convolution operation.

    Parameters
    ----------
    x : TensorLike
        Input array (batch_size, height, width, in_channels)
    w : TensorLike
        Kernel (kh, kw, out_channels, in_channels)
    strides : tp.Union[int, tp.List[int], tp.Tuple[int, int]], optional
        Stride to apply convolution for each axis, by default (1, 1)
    pad : tp.Union[int, tp.List[int], tp.Tuple[int, int]], optional
        Pad width for each axis, by default (0, 0)
    shape : tp.Union[tp.List[int], tp.Tuple[int, int], None], optional
        Output shape, by default None

    Returns
    -------
    Tensor
        Transposed convolution of two arrays

    Examples
    --------
    >>> gd.nn.conv2d_transpose(
    ...     np.arange(16).reshape(1, 4, 4, 1),
    ...     np.arange(9).reshape(3, 3, 1, 1),
    ...     strides=1,
    ...     pad=1,
    ...     shape=(4, 4))
    Tensor([[[[  7.],
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
    return _conv2d_transpose(
        x,
        w.reshape(-1, w.shape[-1]),
        size=w.shape[:2],
        strides=strides,
        pad=pad,
        shape=shape,
    )


class Conv2DTranspose(Module):
    """Two-dimesional tranposed convolution layer.

    Examples
    --------
    >>> c = gd.nn.Conv2DTranspose(1, 10, 3, strides=1, pad=0, bias=True)
    >>> c(np.random.rand(2, 5, 5, 1)).shape
    (2, 7, 7, 10)
    """

    @_typecheck()
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tp.Union[int, tp.List[int], tp.Tuple[int, int]],
        strides: tp.Union[int, tp.List[int], tp.Tuple[int, int]] = (1, 1),
        pad: tp.Union[int, tp.List[int], tp.Tuple[int, int]] = (0, 0),
        shape: tp.Union[tp.List[int], tp.Tuple[int, int], None] = None,
        bias: bool = True,
        dtype: tp.Union[tp.Type[DataType], None] = None,
    ):
        """Initialize 2d transposed convolution module.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : tp.Union[int, tp.List[int], tp.Tuple[int, int]]
            Size of convolution kernel
        strides : tp.Union[int, tp.List[int], tp.Tuple[int, int]], optional
            Strides of kernel convolution, by default (1, 1)
        pad : tp.Union[int, tp.List[int], tp.Tuple[int, int]], optional
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
        self._shape = shape
        self.weight = Tensor(
            np.random.uniform(
                -v, v,
                (kernel_size[0] * kernel_size[1] * out_channels, in_channels)),
            dtype=dtype,
            is_variable=True)
        if bias:
            self.bias = Tensor(
                np.random.uniform(-v, v, out_channels),
                dtype=dtype,
                is_variable=True)

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:  # noqa: D102
        x = _conv2d_transpose(
            x,
            self.weight,
            size=self._kernel_size,
            strides=self._strides,
            pad=self._pad,
            shape=self._shape,
        )
        if hasattr(self, 'bias'):
            x = x + self.bias
        return x
