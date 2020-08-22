import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _Normal(_Operator):

    def __init__(
            self,
            loc,
            scale,
            size: tp.Tuple[int] = None,
            name: str = None):
        super().__init__(loc, scale, name=name)
        self._size = np.broadcast(loc, scale).shape if size is None else size

    def _forward_numpy(self, loc, scale):
        self.eps = np.random.normal(size=self._size)
        return loc + scale * self.eps

    def _backward_numpy(self, delta, loc, scale):
        dloc = _unbroadcast_to(delta, loc.shape)
        dscale = _unbroadcast_to(delta * self.eps, scale.shape)
        return dloc, dscale


@_typecheck(exclude=('loc', 'scale'))
def normal(
        loc: Array,
        scale: Array,
        size: tp.Union[tp.Iterable[int], None] = None,
        *,
        name: str = None) -> Array:
    r"""Return array with normally distributed values.

    .. math::
        \mathcal{N}(x|\mu, \sigma^2) = {1\over\sqrt{2\pi\sigma^2}}\exp\left\{
            -{1\over2\sigma^2}(x-\mu)^2\right\}

    Parameters
    ----------
    loc : Array
        Location parameter of the normal distribution.
    scale : Array
        Scale parameter of the normal distribution.
    size : tp.Union[tp.Iterable[int], None], optional
        Size of the resulting array, by default None
    name : str, optional
        Name of this operation or the resulting array, by default None

    Returns
    -------
    Array
        Array with normally distributed values.

    Examples
    --------
    >>> import pygrad as pg; import numpy as np; np.random.seed(0)
    >>> pg.random.normal(0, 1, (10,))
    array([ 1.76405235,  0.40015721,  0.97873798,  2.2408932 ,  1.86755799,
           -0.97727788,  0.95008842, -0.15135721, -0.10321885,  0.4105985 ])
    >>> pg.random.normal(pg.random.normal(0, 1, [5, 1]), [1, 2], (5, 2))
    array([[0.4777179 , 3.13220172],
           [1.78794783, 4.44243165],
           [1.09471205, 3.74919587],
           [0.45534934, 3.10983316],
           [0.77753756, 3.43202138]])
    """
    if isinstance(loc, Array) or isinstance(scale, Array):
        return _Normal(loc, scale, name=name).forward()
    return Array(
        np.random.normal(loc, scale, size=size),
        name=None if name is None else name + '.out')
