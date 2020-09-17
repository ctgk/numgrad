import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck
from pygrad._utils._unbroadcast import _unbroadcast_to


class _Exponential(_Operator):

    def __init__(self, scale, size: tp.Tuple[int] = None, name: str = None):
        super().__init__(scale, name=name)
        self._size = self._args[0].shape if size is None else size

    def _forward_numpy(self, scale):
        self._eps = np.random.standard_exponential(self._size).astype(
            config.dtype)
        return self._eps * scale

    def _backward_numpy(self, delta, scale):
        return _unbroadcast_to(delta * self._eps, scale.shape)


@_typecheck(exclude_args=('scale',))
def exponential(
        scale: Array,
        size: tp.Union[int, tp.Iterable[int], None] = None,
        *,
        name: str = None) -> Array:
    r"""Return random samples from exponential distribution

    .. math::
        p(x|\beta) = {1\over\beta}e^{-{x\over\beta}}

    Parameters
    ----------
    scale : Array
        Scale parameter of exponential distribution
    size : tp.Union[int, tp.Iterable[int], None], optional
        Size of returned random samples, by default None
    name : str, optional
        The name of the operation, by default None

    Returns
    -------
    Array
        Random samples from the distribution
    """
    size = (size,) if isinstance(size, int) else size
    if isinstance(scale, Array):
        return _Exponential(scale, size, name=name).forward()
    return Array(
        np.random.exponential(scale, size=size),
        name=None if name is None else name + '.out')
