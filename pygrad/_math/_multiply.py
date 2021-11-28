from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _multiply(x: TensorLike, y: TensorLike):
    x_shape = Tensor(x).shape if not isinstance(x, Tensor) else x.shape
    y_shape = Tensor(y).shape if not isinstance(y, Tensor) else y.shape

    def grad(dout):
        dx = _unbroadcast_to(dout * y, x_shape)
        dy = _unbroadcast_to(dout * x, y_shape)
        return dx, dy

    return x * y, grad


def multiply(x: TensorLike, y: TensorLike) -> Tensor:
    """Return element-wise multiplication of two arrays.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    y : TensorLike
        Another tensor-like object.

    Returns
    -------
    Tensor
        Element-wise multiplication of two arrays.

    Examples
    --------
    >>> gd.multiply([[1, 2], [2, 3]], [-1, 3])
    Tensor([[-1.,  6.],
            [-2.,  9.]])
    >>> np.array([[1, 2], [2, 3]]) * gd.Tensor([-1, 3])
    Tensor([[-1.,  6.],
            [-2.,  9.]])
    """
    return _multiply(x, y)
