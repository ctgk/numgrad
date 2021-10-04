import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Dropout(_Operator):

    def __init__(self, x, droprate: float, name: str = None):
        super().__init__(x, name=name)
        self._droprate = droprate

    def _forward_numpy(self, x):
        self._mask = np.random.choice(
            [1 / (1 - self._droprate), 0],
            size=x.shape,
            p=[1 - self._droprate, self._droprate])
        if 'float' in repr(x.dtype):
            self._mask = self._mask.astype(x.dtype)
        return x * self._mask

    def _backward_numpy(self, delta, x):
        return delta * self._mask


@_typecheck(exclude_args=('x',))
def dropout(x: Array,
            droprate: tp.Union[float, None] = 0.5,
            *,
            name: str = None) -> Array:
    """Element-wise dropout function.

    Parameters
    ----------
    x : Array
        Input array
    droprate : tp.Union[float, None], optional
        Probability of dropping values, by default 0.5
    name : str, optional
        The name of the operation, by default None

    Returns
    -------
    Array
        The output of dropout function

    Examples
    --------
    >>> import pygrad as gd; import numpy as np; np.random.seed(1111)
    >>> gd.nn.dropout([1, -2, 3, -4])
    array([ 2., -0.,  6., -8.])
    """
    if droprate is None:
        return x
    return _Dropout(x, droprate, name=name).forward()


class Dropout(Module):
    """Dropout module."""

    @_typecheck()
    def __init__(self, droprate: float = 0.5):
        """Initialize dropout module.

        Parameters
        ----------
        droprate : float, optional
            Rate of dropping values, by default 0.5
        """
        super().__init__()
        self._droprate = droprate

    def __call__(self, x: Array, **kwargs) -> Array:
        """Return output of dropout layer.

        Parameters
        ----------
        x : Array
            Input

        Returns
        -------
        Array
            Output of dropout layer.
        """
        return dropout(x, droprate=kwargs.get('droprate'))
