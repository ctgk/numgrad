import typing as tp

import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Min(_Operator):

    def __init__(
            self,
            x,
            axis: tp.Union[int, tp.Iterable[int]] = None,
            keepdims: bool = False,
            name=None):
        super().__init__(x, name=name)
        self._axis = (axis,) if isinstance(axis, int) else axis
        self._keepdims = keepdims

    def _forward_numpy(self, x):
        return np.min(x, axis=self._axis, keepdims=self._keepdims)

    def _backward_numpy(self, dy, x):
        if all((isinstance(dy, np.ndarray),
                (not self._keepdims),
                (self._axis is not None))):
            axis_positive = []
            for axis in self._axis:
                axis_positive.append(x.ndim + axis if axis < 0 else axis)
            for axis in sorted(axis_positive):
                dy = np.expand_dims(dy, axis)
        dx = 1 * np.broadcast_to(dy, x.shape)
        dx[np.where(x != x.min(axis=self._axis, keepdims=True))] = 0
        return dx


@_typecheck(exclude=('x',))
def min(x: Array,
        axis: tp.Union[int, tp.Iterable[int]] = None,
        keepdims: bool = False,
        *,
        name: str = None) -> Array:
    """Return minimum element along given axis

    Parameters
    ----------
    x : Array
        Input array.
    axis : tp.Union[int, tp.Iterable[int]], optional
        Axis to find minimum value along, by default None
    keepdims : bool, optional
        Whether to keep dimensionality of the array or not, by default False
    name : str, optional
        Name of the operation, by default None

    Returns
    -------
    Array
        Minimum element.

    Examples
    --------
    >>> import pygrad as pg
    >>> a = pg.Array([[2, 3], [-1, -9]])
    >>> a.min()
    array(-9)
    >>> a.min(axis=0)
    array([-1, -9])
    >>> a.min(axis=1, keepdims=True)
    array([[ 2],
           [-9]])
    """
    return _Min(x, axis, keepdims, name=name).forward()
