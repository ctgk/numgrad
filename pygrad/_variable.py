import typing as tp

import numpy as np

from pygrad._config import config


ArrayLike = tp.Union[np.ndarray, list, tuple]


def _ndarray_views(*args):
    return [a.view(np.ndarray) if isinstance(a, Variable) else a for a in args]


class Variable(np.ndarray):
    """Multi-dimensional variable class.

    Examples
    --------
    >>> a = gd.Variable([0, 1])
    >>>
    >>> # data type
    >>> a.dtype
    dtype('float64')
    >>> gd.Variable([1], dtype=np.float32).dtype
    dtype('float32')
    >>>
    >>> # numpy ufunc
    >>> b = a + 1
    >>> type(b)
    <class 'pygrad.Variable'>
    >>> b
    Variable([1., 2.])
    """

    def __new__(cls, array: ArrayLike, dtype: type = None) -> 'Variable':
        """Return tensor.

        Parameters
        ----------
        array : ArrayLike
            Input array
        dtype : type, optional
            One of {`float`, `np.float32`, `np.float64`}, by default None

        Returns
        -------
        Variable
            Output variable

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
        return np.asarray(array, dtype=dtype).view(Variable)

    def __array_ufunc__(  # noqa: D105
        self, ufunc, method, *inputs, out=None, **kwargs,
    ):
        if ufunc.nout != 1:
            raise NotImplementedError
        if method != '__call__':
            raise NotImplementedError

        args = _ndarray_views(*inputs)
        if out:
            kwargs['out'] = _ndarray_views(*out)[0]
        result = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        result = np.asarray(result).view(Variable)
        if config._graph is not None:
            config._graph._add_node(result, ufunc, *inputs, **kwargs)
        return result


VariableLike = tp.Union[Variable, ArrayLike]
