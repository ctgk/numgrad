import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Transpose(_Operator):

    def __init__(self, x, axes: tp.Iterable[int], name: str = None):
        super().__init__(x, name=name)
        self._axes = axes

    def _forward_numpy(self, x):
        return np.transpose(x, self._axes)

    def _backward_numpy(self, delta, x):
        if self._axes is None:
            return np.transpose(delta)
        return np.transpose(delta, np.argsort(self._axes))


@_typecheck()
def transpose(x, axes: tp.Iterable[int] = None, *, name: str = None) -> Array:
    """Return a transposed array.

    Parameters
    ----------
    x : Array
        Input array.
    axes : tp.Iterable[int], optional
        New shape
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Transposed array.

    Examples
    --------
    >>> import pygrad as pg
    >>> pg.transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    array([[1, 4, 7],
           [2, 5, 8],
           [3, 6, 9]])
    >>> pg.transpose([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], (1, 2, 0))
    array([[[1],
            [2],
            [3]],
    <BLANKLINE>
           [[4],
            [5],
            [6]],
    <BLANKLINE>
           [[7],
            [8],
            [9]]])
    """
    return _Transpose(x, axes, name=name).forward()
