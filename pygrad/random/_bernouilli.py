import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._errors import DifferentiationError
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Bernoulli(_Operator):

    def __init__(self, p, size: tp.Tuple[int] = None, name=None):
        super().__init__(p, name=name)
        self._differentiable = False
        self._size = p.shape if size is None else size

    def _forward_numpy(self, p):
        return (np.random.uniform(size=self._size) < p).astype(np.int)

    def _backward_numpy(self, *args, **kwargs):
        raise DifferentiationError(
            'pygrad.random.bernoulli() is not a differentiable operation.')


@_typecheck(exclude_args=('p',))
def bernoulli(
        p: Array,
        size: tp.Union[tp.Iterable[int], None] = None,
        *,
        name: str = None):
    """Return array with values following bernoulli distributions.

    Parameters
    ----------
    p : Array
        Probability of positive.
    size : tp.Union[tp.Iterable[int], None], optional
        The output shape, by default None
    name : str, optional
        The name of the operation, by default None

    Returns
    -------
    Array
        Array with values following bernoulli distributions

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(0)
    >>> gd.random.bernoulli([0, 0.8, 1], (5, 3))
    array([[0, 1, 1],
           [0, 1, 1],
           [0, 0, 1],
           [0, 1, 1],
           [0, 0, 1]])
    """
    return _Bernoulli(p, size, name).forward()
