from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


@_typecheck()
@differentiable_operator
def _add(x: TensorLike, y: TensorLike):
    x_shape = Tensor(x).shape if not isinstance(x, Tensor) else x.shape
    y_shape = Tensor(y).shape if not isinstance(y, Tensor) else y.shape

    def grad(dout):
        dx = _unbroadcast_to(dout, x_shape)
        dy = _unbroadcast_to(dout, y_shape)
        return dx, dy

    return x + y, grad


def add(x: TensorLike, y: TensorLike) -> Tensor:
    """Return element-wise addition of two arrays.

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    y : TensorLike
        Another input tensor-like object.

    Returns
    -------
    Tensor
        Element-wise addition of two arrays.

    Examples
    --------
    >>> a = gd.Tensor([[1, 2], [2, 3]])
    >>> b = np.array([-1, 3])
    >>> gd.add(a, b)
    Tensor([[0., 5.],
            [1., 6.]])
    >>> a + b
    Tensor([[0., 5.],
            [1., 6.]])
    """
    return _add(x, y)
