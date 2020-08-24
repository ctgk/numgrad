import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _Uniform(_Operator):

    def __init__(self, low, high, size: tp.Tuple[int] = None, name=None):
        super().__init__(low, high, name=name)
        self._size = np.broadcast(low, high).shape if size is None else size

    def _forward_numpy(self, low, high):
        self._u = np.random.uniform(0, 1, size=self._size)
        return low + (high - low) * self._u

    def _backward_numpy(self, delta, low, high):
        du = delta * self._u
        dmin = _unbroadcast_to(delta - du, low.shape)
        dmax = _unbroadcast_to(du, high.shape)
        return dmin, dmax


@_typecheck(exclude_args=('low', 'high'))
def uniform(low: Array,
            high: Array,
            size: tp.Union[tp.Iterable[int], None] = None,
            *,
            name: str = None) -> Array:
    r"""Return array with uniformly distributed values.

    .. math::
        \mathcal{U}(x|a, b) = \begin{cases}
            {1\over b - a} & a \le x \le b\\
            0 & {\rm otherwise}
        \end{cases}

    Parameters
    ----------
    low : Array
        Lower boundary of the output interval.
    high : Array
        Upper boundary of the output interval.
    size : tp.Union[tp.Iterable[int], None], optional
        Output shape, by default None
    name : str, optional
        Name of this operation or the output array, by default None

    Returns
    -------
    Array
        Array with uniformly distributed values.

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(0)
    >>> gd.random.uniform(0, 1, (10,))
    array([0.5488135 , 0.71518937, 0.60276338, 0.54488318, 0.4236548 ,
           0.64589411, 0.43758721, 0.891773  , 0.96366276, 0.38344152])
    >>> gd.random.uniform([1, -2], [2, 0], size=(5, 2))
    array([[ 1.79172504, -0.94221016],
           [ 1.56804456, -0.14880672],
           [ 1.07103606, -1.8257414 ],
           [ 1.0202184 , -0.33476031],
           [ 1.77815675, -0.2599757 ]])
    """
    if isinstance(low, Array) or isinstance(high, Array):
        return _Uniform(low, high, size=size, name=name).forward()
    return Array(
        np.random.uniform(low, high, size),
        name=None if name is None else name + '.out')
