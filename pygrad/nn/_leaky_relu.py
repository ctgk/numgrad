import numpy as np

from pygrad._core._differentiable_operator import differentiable_operator
from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._utils._typecheck import _typecheck


@_typecheck()
@differentiable_operator
def _leaky_relu(x: TensorLike, *, alpha: float = 0.2):

    def grad(dout):
        return ((x > 0) + alpha * (x <= 0)) * dout

    out = np.maximum(x, 0) + np.minimum(x, 0) * alpha
    return out, grad


def leaky_relu(x: TensorLike, alpha: float = 0.2) -> Tensor:
    r"""Element-wise leaky rectified linear unit.

    .. math::
        {\rm LeakyReLU}(x) = \begin{cases}
            x & (x > 0)\\
            \alpha x & (x \le 0)\\
            \end{cases}

    Parameters
    ----------
    x : TensorLike
        Input tensor-like object.
    alpha : float, optional
        Coefficient of leakage, by default 0.2

    Returns
    -------
    Tensor
        The output of leaky rectified linear unit.

    Examples
    --------
    >>> gd.nn.leaky_relu([1, -1, 2, -3])
    Tensor([ 1. , -0.2,  2. , -0.6])
    """
    return _leaky_relu(x, alpha=alpha)


class LeakyReLU(Module):
    """Leaky rectified linear unit layer."""

    @_typecheck()
    def __init__(self, alpha: float = 0.2):
        """Initialize leaky ReLU module.

        Parameters
        ----------
        alpha : float, optional
            Coefficient of leak, by default 0.2
        """
        super().__init__()
        self._alpha = alpha

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:
        """Return result of leaky ReLU function.

        Parameters
        ----------
        x : TensorLike
            Input.

        Returns
        -------
        Tensor
            Result of leaky ReLU.
        """
        return _leaky_relu(x, alpha=self._alpha)
