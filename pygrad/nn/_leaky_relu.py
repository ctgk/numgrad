import numpy as np

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _LeakyReLU(_Operator):

    def __init__(self, x, alpha: float = 0.2, name: str = None):
        super().__init__(x, name=name)
        self._alpha = alpha

    def _forward_numpy(self, x):
        return np.maximum(x, 0) + np.minimum(x, 0) * self._alpha

    def _backward_numpy(self, delta, x):
        return delta * ((x > 0) + self._alpha * (x <= 0))


@_typecheck(exclude_args=('x',))
def leaky_relu(x: Array, alpha: float = 0.2, *, name: str = None) -> Array:
    r"""Element-wise leaky rectified linear unit.

    .. math::
        {\rm LeakyReLU}(x) = \begin{cases}
            x & (x > 0)\\
            \alpha x & (x \le 0)\\
            \end{cases}

    Parameters
    ----------
    x : Array
        Input array.
    alpha : float, optional
        Coefficient of leakage, by default 0.2
    name : str, optional
        The name of the operation, by default None

    Returns
    -------
    Array
        The output of leaky rectified linear unit.

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.nn.leaky_relu([1, -1, 2, -3])
    array([ 1. , -0.2,  2. , -0.6])
    """
    return _LeakyReLU(x, alpha, name=name).forward()


class LeakyReLU(Module):
    """Leaky rectified linear unit layer
    """

    @_typecheck()
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self._alpha = alpha

    def __call__(self, x: Array, **kwargs) -> Array:
        return _LeakyReLU(x, self._alpha).forward()
