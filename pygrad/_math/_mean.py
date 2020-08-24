from typing import Iterable, Union

import numpy as np

from pygrad._core._array import Array
from pygrad._utils._typecheck import _typecheck
from pygrad._math._sum import _Sum


class _Mean(_Sum):

    def __init__(self, x, axis=None, keepdims=False, name=None):
        super().__init__(x, axis=axis, keepdims=keepdims, name=name)

    def _forward_numpy(self, x):
        return np.mean(x, axis=self.axis, keepdims=self.keepdims)

    def _backward_numpy(self, dy, x):
        return super()._backward_numpy(dy, x) * dy.size / x.size


@_typecheck(exclude_args=('x',))
def mean(
        x: Array,
        axis: Union[int, Iterable[int], None] = None,
        keepdims: bool = False,
        *,
        name: str = None) -> Array:
    """Mean of elements in the array along given axis.

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
        Mean of the input array.

    Examples
    --------
    >>> import pygrad as gd
    >>> gd.Array([1, 2, 3]).mean()
    array(2.)
    >>> gd.mean(gd.Array([[1., 2.], [4., 8.]]), axis=1, keepdims=True)
    array([[1.5],
           [6. ]])
    """
    return _Mean(x, axis, keepdims, name=name).forward()
