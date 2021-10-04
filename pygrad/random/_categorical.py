import typing as tp

import numpy as np
import scipy.special as sp

from pygrad._core._array import Array
from pygrad._core._errors import DifferentiationError
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Categorical(_Operator):

    def __init__(
            self,
            logits,
            axis: int = -1,
            size: tp.Tuple[int] = None,
            name: str = None):
        super().__init__(logits, name=name)
        self._differentiable = False
        self._axis = axis
        self._size = size

    def _forward_numpy(self, logits):
        g = np.random.gumbel(
            size=logits.shape if self._size is None else self._size)
        v = sp.log_softmax(logits, self._axis) + g
        m = np.max(v, self._axis, keepdims=True)
        return (v == m).astype(logits.dtype)

    def _backward_numpy(self, delta, logits):
        raise DifferentiationError(
            'pygrad.random.categorical() is not a differentiable operation.')


@_typecheck(exclude_types=(Array,))
def categorical(
        logits: Array,
        axis: int = -1,
        size: tp.Union[tp.Iterable[int], None] = None,
        *,
        name: tp.Union[str, None] = None) -> Array:
    """Return array whose values follow categorical distribution.

    Parameters
    ----------
    logits : Array
        Logits of probabilities along the given axis
    axis : int, optional
        Axis specifying probability distributions, by default -1
    size : tp.Union[tp.Iterable[int], None], optional
        The output shape, by default None
    name : tp.Union[str, None], optional
        The name of the operation, by default None

    Returns
    -------
    Array
        Random sample from categorical distribution

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(0)
    >>> gd.random.categorical([0, -1, 1], size=(5, 3))
    array([[0., 0., 1.],
           [0., 0., 1.],
           [1., 0., 0.],
           [0., 0., 1.],
           [0., 0., 1.]])
    """
    return _Categorical(logits, axis=axis, size=size, name=name).forward()
