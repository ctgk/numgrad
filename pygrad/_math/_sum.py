from typing import Iterable, Union

import numpy as np

from pygrad._core._array import Array
from pygrad._core._operator import _Operator
from pygrad._utils._typecheck import _typecheck


class _Sum(_Operator):

    def __init__(
            self,
            x: Array,
            axis: Union[int, Iterable[int], None] = None,
            keepdims: bool = False,
            name: str = None):
        super().__init__(x, name=name)
        if isinstance(axis, int):
            axis = (axis,)
        self.axis = axis
        self.keepdims = keepdims

    def _forward_numpy(self, x):
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)

    def _backward_numpy(self, dy, x):
        if all((
            isinstance(dy, np.ndarray),
            (not self.keepdims),
            (self.axis is not None),
        )):
            axis_positive = []
            for axis in self.axis:
                if axis < 0:
                    axis_positive.append(x.ndim + axis)
                else:
                    axis_positive.append(axis)
            for axis in sorted(axis_positive):
                dy = np.expand_dims(dy, axis)
        dx = np.broadcast_to(dy, x.shape)
        return dx


@_typecheck(exclude_args=('x',))
def sum(
    x: Array,
    axis: Union[int, Iterable[int], None] = None,
    keepdims: bool = False,
    *,
    name: str = None,
) -> Array:
    """Sum elements in the array along given axis.

    Parameters
    ----------
    x : Array
        Input array.
    axis : Union[int, Iterable[int], None], optional
        Axis to sum along, by default None
    keepdims : bool, optional
        Whether to keep dimensionality of the array or not, by default False
    name : str, optional
        Name of the operation, by default None.

    Returns
    -------
    Array
        Summation.

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.Array([1, 2, 3]).sum()
    array(6)
    >>> gd.sum(gd.Array([[1, 2], [4, 8]]), axis=1, keepdims=True)
    array([[ 3],
           [12]])
    """
    return _Sum(x, axis, keepdims, name=name).forward()
