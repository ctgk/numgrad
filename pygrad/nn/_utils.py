import itertools
import typing as tp

import numpy as np
from numpy.lib.stride_tricks import as_strided


def _im2col(
        im: np.ndarray,
        size: tp.Iterable[int],
        step: tp.Iterable[int] = (1, 1)) -> np.ndarray:
    """Image to columns.

    Examples
    --------
    >>> import numpy as np
    >>> im = np.arange(16).reshape(1, 4, 4, 1)
    >>> np.squeeze(im)
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> col = _im2col(im, (2, 2))
    >>> col.shape
    (1, 3, 3, 2, 2, 1)
    >>> np.squeeze(col)
    array([[[[ 0,  1],
             [ 4,  5]],
    <BLANKLINE>
            [[ 1,  2],
             [ 5,  6]],
    <BLANKLINE>
            [[ 2,  3],
             [ 6,  7]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[ 4,  5],
             [ 8,  9]],
    <BLANKLINE>
            [[ 5,  6],
             [ 9, 10]],
    <BLANKLINE>
            [[ 6,  7],
             [10, 11]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[ 8,  9],
             [12, 13]],
    <BLANKLINE>
            [[ 9, 10],
             [13, 14]],
    <BLANKLINE>
            [[10, 11],
             [14, 15]]]])
    """
    assert im.ndim == 4
    assert len(size) == 2
    assert len(step) == 2
    slices = tuple(slice(None, None, s) for s in step)
    strides = im[(slice(None),) + slices].strides[:-1] + im.strides[1:]
    out_shape = tuple(np.subtract(im.shape[1: -1], size) // np.array(step) + 1)
    out_shape = (len(im),) + out_shape + size + (np.size(im, -1),)
    patch = as_strided(im, shape=out_shape, strides=strides)
    return patch


def _col2im(col: np.ndarray, step: tp.Iterable[int], out: np.ndarray):
    """Columns to image.

    Examples
    --------
    >>> import numpy as np
    >>> col = _im2col(np.arange(16).reshape(1, 4, 4, 1), (2, 2))
    >>> col.shape
    (1, 3, 3, 2, 2, 1)
    >>> im = np.zeros((1, 4, 4, 1))
    >>> _col2im(col, step=(1, 1), out=im)
    >>> np.squeeze(im)
    array([[ 0.,  2.,  4.,  3.],
           [ 8., 20., 24., 14.],
           [16., 36., 40., 22.],
           [12., 26., 28., 15.]])
    """
    kx, ky = col.shape[3:5]
    col_ = _im2col(out, (kx, ky), step)
    for i, j in itertools.product(range(kx), range(ky)):
        col_[..., i, j, :] += col[..., i, j, :]


def _to_pair(
        obj: tp.Union[int, tp.Iterable[int]], name: str) -> tp.Iterable[int]:
    if isinstance(obj, int):
        return (obj, obj)
    if len(obj) != 2:
        raise ValueError(f'Length of arg \'{name}\' must be 2, not {len(obj)}')
    return obj
