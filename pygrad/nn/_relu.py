import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _relu(x: TensorLike):

    def grad(dout):
        return (x > 0) * dout

    return np.maximum(x, 0), grad


def relu(x: TensorLike) -> Tensor:
    r"""Element-wise rectified linear unit.

    .. math::
        {\rm ReLU}(x) = \max(x, 0)

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.

    Returns
    -------
    Tensor
        The output of rectified linear unit.

    Examples
    --------
    >>> gd.nn.relu([1, -1, 2, -3])
    Tensor([1., 0., 2., 0.])
    """
    return _relu(x)


class ReLU(Module):
    """Rectified linear unit layer."""

    def __init__(self):
        """Initialize ReLU module."""
        super().__init__()

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:
        """Pass through ReLU.

        Parameters
        ----------
        x : TensorLike
            Input

        Returns
        -------
        Tensor
            Activation.
        """
        return _relu(x)
