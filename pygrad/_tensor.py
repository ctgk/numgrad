import typing as tp

import numpy as np

from pygrad._config import config


ArrayLike = tp.Union[np.ndarray, list, tuple]


class Tensor(np.ndarray):
    """Multi-dimensional tensor class.

    Examples
    --------
    >>> a = gd.Tensor([0, 1])
    >>>
    >>> # data type
    >>> a.dtype
    dtype('float64')
    >>> gd.Tensor([1], dtype=np.float32).dtype
    dtype('float32')
    >>>
    >>> # numpy ufunc
    >>> b = a + 1
    >>> type(b)
    <class 'pygrad.Tensor'>
    >>> b
    Tensor([1., 2.])
    """

    def __new__(cls, array: ArrayLike, dtype: type = None) -> 'Tensor':
        """Return tensor.

        Parameters
        ----------
        array : ArrayLike
            Input array
        dtype : type, optional
            One of {`float`, `np.float32`, `np.float64`}, by default None

        Returns
        -------
        Tensor
            Output tensor

        Raises
        ------
        ValueError
            Invalid data type.
        """
        if dtype is None:
            dtype = config.dtype
        if dtype not in (float, np.float32, np.float64):
            raise ValueError(
                'Data type of `Tensor` must be either '
                '`float`, `np.float32`, or `np.float64`, '
                f'not {dtype}')
        return np.asarray(array, dtype=dtype).view(Tensor)

    @staticmethod
    def _ndarray_views(*args):
        return [
            a.view(np.ndarray) if isinstance(a, Tensor) else a
            for a in args
        ]

    def __array_ufunc__(  # noqa: D105
        self, ufunc, method, *inputs, out=None, **kwargs,
    ):
        if ufunc.nout != 1:
            raise NotImplementedError
        if method != '__call__':
            raise NotImplementedError

        args = self._ndarray_views(*inputs)
        if out:
            kwargs['out'] = self._ndarray_views(*out)[0]
        result = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        result = np.asarray(result).view(Tensor)
        if config._graph is not None:
            config._graph._add_node(result, ufunc, *inputs, **kwargs)
        return result


TensorLike = tp.Union[Tensor, ArrayLike]
