import scipy.special as sp

from pygrad._array import Array
from pygrad._operator import _Operator
from pygrad._type_check import _typecheck_args


class _Gamma(_Operator):

    def __init__(self, x, name=None):
        super().__init__(x, name=name)

    def _forward_numpy(self, x):
        self.output = sp.gamma(x)
        return self.output

    def _backward_numpy(self, delta, x):
        return delta * sp.digamma(x) * self.output


@_typecheck_args
def gamma(x, *, name: str = None) -> Array:
    r"""Element-wise gamma function.

    .. math::
        \Gamma(z) &= \int^{\infty}_0 e^{-t}t^{z-1}{\rm d}t

        {{\rm d}\over{\rm d} z}\Gamma(z) &= \psi(z)\Gamma(z),

    where :math:`\psi(z)={{\rm d}\over{\rm d} z}\ln\Gamma(z)`.

    Parameters
    ----------
    x
        Input array.
    name : str, optional
        Name of this operation, by default None

    Returns
    -------
    Array
        Value of element-wise gamma function.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.gamma([1, 2, 3, 4])
    array([1., 1., 2., 6.])
    """
    return _Gamma(x, name=name).forward()
