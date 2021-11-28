import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _matrix_vector_multiply(x: TensorLike, y: TensorLike):
    def grad(dout):
        dout = np.expand_dims(dout, -1)
        dout = np.broadcast_to(dout, x.shape)
        dx = dout * y
        dy = _unbroadcast_to(dout * x, y.shape)
        return dx, dy
    return x @ y, grad


@_typecheck()
@differentiable_operator
def _vector_matrix_multiply(x: TensorLike, y: TensorLike):
    def grad(dout):
        dout = np.expand_dims(dout, -2)
        dout = np.broadcast_to(dout, y.shape)
        dx = _unbroadcast_to((dout * y).sum(axis=-1), x.shape)
        dy = dout * x[:, None]
        return dx, dy
    return x @ y, grad


@_typecheck()
@differentiable_operator
def _matrix_multiply(x: TensorLike, y: TensorLike):
    def grad(dout):
        dx = dout @ y.T
        dy = x.T @ dout
        return dx, dy
    return x @ y, grad


@_typecheck()
@differentiable_operator
def _batch_matrix_multiply(x: TensorLike, y: TensorLike):
    def grad(dout):
        dx = _unbroadcast_to(dout @ np.swapaxes(y, -1, -2), x.shape)
        dy = _unbroadcast_to(np.swapaxes(x, -1, -2) @ dout, y.shape)
        return dx, dy
    return x @ y, grad


def matmul(x: TensorLike, y: TensorLike) -> Tensor:
    """Return matrix multiplication of two arrays.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    y : TensorLike
        Another tensor-like object.

    Returns
    -------
    Tensor
        matrix multiplication of two arrays.

    Examples
    --------
    >>> gd.matmul([[1, 2], [2, 3]], [-1, 3])
    Tensor([5., 7.])
    >>> np.array([-1, 3]) @ gd.Tensor([[1, 2], [2, 3]])
    Tensor([5., 7.])
    >>> gd.matmul([[1, 2], [2, 3]], [[-1], [3]])
    Tensor([[5.],
            [7.]])
    >>> gd.matmul([[[1, 2], [2, 3]], [[-1, 2], [2, -3]]], [[-1], [3]])
    Tensor([[[  5.],
             [  7.]],
    <BLANKLINE>
            [[  7.],
             [-11.]]])
    """
    x_ndim = x.ndim if isinstance(x, Tensor) else Tensor(x).ndim
    y_ndim = y.ndim if isinstance(y, Tensor) else Tensor(y).ndim

    if x_ndim == 1:
        return _vector_matrix_multiply(x, y)
    if y_ndim == 1:
        return _matrix_vector_multiply(x, y)
    if x_ndim == y_ndim == 2:
        return _matrix_multiply(x, y)
    return _batch_matrix_multiply(x, y)
