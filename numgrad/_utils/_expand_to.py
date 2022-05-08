import typing as tp

import numpy as np

from numgrad._variable import Variable


def _expand_to(
    a: tp.Union[np.number, np.ndarray, Variable],
    ndim_or_shape: tp.Union[int, tp.Tuple[int, ...]],
    axis: tp.Union[None, int, tp.Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> tp.Union[np.ndarray, Variable]:
    """Return expanded array.

    Parameters
    ----------
    a : tp.Tuple[np.number, np.ndarray, Variable]
        Input array-like object to expand.
    ndim_or_shape : tp.Union[int, tp.Tuple[int, ...]]
        Target ndim (int) or shape (tuple) to expand to.
    axis : tp.Union[None, int, tp.Tuple[int, ...]], optional
        Argument used in reduction function, by default None
    keepdims : bool, optional
        Another argument used in reduction function, by default False

    Returns
    -------
    array
        Expanded array or scalar.
    """
    is_ndim = isinstance(ndim_or_shape, int)
    ndim = ndim_or_shape if is_ndim else len(ndim_or_shape)
    shape = None if is_ndim else ndim_or_shape

    if (is_ndim and a.ndim == ndim) or a.shape == shape:
        return a
    if (not keepdims) and (axis is not None):
        axis_positive = []
        for ax in axis if isinstance(axis, tuple) else (axis,):
            if ax < 0:
                axis_positive.append(ndim + ax)
            else:
                axis_positive.append(ax)
        for ax in sorted(axis_positive):
            a = np.expand_dims(a, ax)
    if is_ndim:
        return a
    return np.broadcast_to(a, shape)


def _expand_to_if_array(
    a: tp.Union[np.number, np.ndarray, Variable],
    ndim_or_shape: tp.Union[int, tp.Tuple[int, ...]],
    axis: tp.Union[None, int, tp.Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> tp.Union[np.number, np.ndarray, Variable]:
    """Return expanded array.

    Parameters
    ----------
    a : tp.Tuple[np.number, np.ndarray, Variable]
        Input array-like object to expand. If this is scalar, then returns
        the input as is.
    ndim_or_shape : tp.Union[int, tp.Tuple[int, ...]]
        Target ndim (int) or shape (tuple) to expand to.
    axis : tp.Union[None, int, tp.Tuple[int, ...]], optional
        Argument used in reduction function, by default None
    keepdims : bool, optional
        Another argument used in reduction function, by default False

    Returns
    -------
    array
        Expanded array or scalar.
    """
    if a.ndim == 0:
        return a
    return _expand_to(a, ndim_or_shape, axis, keepdims)
