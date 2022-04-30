import typing as tp

import numpy as np

from pygrad._config import config


ArrayLike = tp.Union[np.ndarray, list, tuple]


def _ndarray_args(*args):
    return tuple(
        a.view(np.ndarray) if isinstance(a, Variable) else a for a in args)


def _ndarray_kwargs(**kwargs):
    return {
        k: (v.view(np.ndarray) if isinstance(v, Variable) else v)
        for k, v in kwargs.items()
    }


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
    <class 'numpy.ndarray'>
    >>> b
    array([1., 2.])
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
        if config._verbosity > 0:
            print(ufunc, method, inputs, out, kwargs)
        if ufunc.nout != 1:
            raise NotImplementedError
        if method not in ('__call__', 'reduce'):
            raise NotImplementedError

        if out:
            kwargs['out'] = _ndarray_args(*out)[0]
        result = super().__array_ufunc__(
            ufunc, method,
            *_ndarray_args(*inputs), **_ndarray_kwargs(**kwargs),
        )
        if result is NotImplemented:
            return NotImplemented
        if config._graph is not None:
            result = Variable(result)
            config._graph._add_node(
                result,
                ufunc if method == '__call__' else getattr(ufunc, method),
                *inputs, **kwargs,
            )
        return result


for method in ('mean',):
    setattr(
        Variable, method,
        lambda self, *args, **kwargs: eval(f'np.{method}')(
            self.view(np.ndarray) if config._graph is None else self,
            *args, **kwargs,
        ),
    )
    setattr(
        getattr(Variable, method),
        '__doc__',
        eval(f'np.ndarray.{method}').__doc__,
    )


Variable.transpose = lambda self, *axes: np.transpose(
    self.view(np.ndarray) if config._graph is None else self,
    *({0: tuple(), 1: axes}.get(len(axes), (axes,))),
)
Variable.transpose.__doc__ = np.ndarray.transpose.__doc__
Variable.T = property(lambda self: self.transpose())


VariableLike = tp.Union[Variable, ArrayLike]
