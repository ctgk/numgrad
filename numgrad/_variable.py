import functools

import numpy as np
import numpy.typing as npt

from numgrad._config import config


def _ndarray_args(*args):
    return tuple(
        a._data if isinstance(a, Variable) else a for a in args)


def _ndarray_kwargs(**kwargs):
    return {
        k: (v._data if isinstance(v, Variable) else v)
        for k, v in kwargs.items()
    }


class Variable(object):
    """Multi-dimensional variable class.

    Examples
    --------
    >>> a = ng.Variable([0, 1])
    >>>
    >>> a
    Variable([0., 1.])
    >>> # data type
    >>> a.dtype
    dtype('float64')
    >>> ng.Variable([1], dtype=np.float32).dtype
    dtype('float32')
    >>>
    >>> # numpy ufunc
    >>> b = a + 1
    >>> type(b)
    <class 'numpy.ndarray'>
    >>> b
    array([1., 2.])
    """

    def __init__(self, data: npt.ArrayLike, dtype: npt.DTypeLike = None):
        """Construct variable object to compute gradient with respect to.

        Parameters
        ----------
        data : npt.ArrayLike
            Input data
        dtype : npt.DTypeLike, optional
            Data type which must be either np.float32 or np.float64,
            by default None.
        """
        if dtype is None:
            dtype = config.dtype
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                'Data type of `Variable` must be either '
                '`np.float32`, or `np.float64`, '
                f'not {dtype}')
        if np.isscalar(data):
            self._data = dtype(data)
        else:
            self._data = np.asarray(data, dtype=dtype)

    def __array_ufunc__(  # noqa: D105
        self, ufunc, method, *inputs, out=None, **kwargs,
    ):
        if config._verbosity > 0:
            print(
                'inputs of __array_ufunc__',
                ufunc, method, inputs, out, kwargs,
            )
        if ufunc.nout != 1:
            raise NotImplementedError
        if method not in ('__call__', 'reduce'):
            raise NotImplementedError

        if out:
            kwargs['out'] = _ndarray_args(*out)[0]
        result = getattr(ufunc, method)(
            *_ndarray_args(*inputs), **_ndarray_kwargs(**kwargs))
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

    def __array_function__(self, func, types, args, kwargs):  # noqa: D105
        # https://numpy.org/devdocs/user/basics.dispatch.html
        if config._verbosity > 0:
            print('inputs of __array_function__:', func, types, args, kwargs)
        result = func(*_ndarray_args(*args), **_ndarray_kwargs(**kwargs))
        if config._graph is not None:
            result = Variable(result)
            config._graph._add_node(result, func, *args, **kwargs)
        return result


for property_, func in (
    ('dtype', property(lambda self: getattr(self._data, 'dtype'))),
    ('T', property(lambda self: getattr(np, 'transpose')(self))),
):
    setattr(Variable, property_, func)


def _inplace(self, inplace_op, other):
    if config._graph is None:
        getattr(self._data, inplace_op)(other)
        return self
    else:
        raise ValueError(
            'Computation graph does not support inplace operations')


for method, func in (
    (
        '__array__',
        lambda self, dtype=None: np.asarray(self._data, dtype=dtype),
    ),
    ('__float__', lambda self: float(self._data)),
    ('__len__', lambda self: len(self._data)),
    (
        '__repr__',
        lambda self: repr(self._data.view(
            type('Variable', (np.ndarray,), {}))),
    ),
    ('__iadd__', functools.partialmethod(_inplace, '__iadd__')),
    ('__isub__', functools.partialmethod(_inplace, '__isub__')),
    ('__imul__', functools.partialmethod(_inplace, '__imul__')),
    ('__itruediv__', functools.partialmethod(_inplace, '__itruediv__')),

    # differentiable operations
    ('__pos__', lambda self: getattr(np, 'positive')(self)),
    ('__neg__', lambda self: getattr(np, 'negative')(self)),
    ('__add__', lambda a, b: getattr(np, 'add')(a, b)),
    ('__radd__', lambda a, b: getattr(np, 'add')(a, b)),
    ('__sub__', lambda a, b: getattr(np, 'subtract')(a, b)),
    ('__rsub__', lambda a, b: getattr(np, 'subtract')(b, a)),
    ('__mul__', lambda a, b: getattr(np, 'multiply')(a, b)),
    ('__rmul__', lambda a, b: getattr(np, 'multiply')(a, b)),
    ('__truediv__', lambda a, b: getattr(np, 'divide')(a, b)),
    ('__rtruediv__', lambda a, b: getattr(np, 'divide')(b, a)),
    ('__matmul__', lambda a, b: getattr(np, 'matmul')(a, b)),
    ('__rmatmul__', lambda a, b: getattr(np, 'matmul')(b, a)),
    (
        'reshape',
        lambda a, *args, **kwargs: getattr(np, 'reshape')(
            a, *(args if len(args) == 1 else (args,)), **kwargs),
    ),
    (
        'transpose',
        lambda a, *args, **kwargs: getattr(np, 'transpose')(
            a, *({0: tuple(), 1: args}.get(len(args), (args,))), **kwargs),
    ),
    (
        'max',
        lambda self, *args, **kwargs: getattr(np, 'max')(
            self, *args, **kwargs),
    ),
    (
        'min',
        lambda self, *args, **kwargs: getattr(np, 'min')(
            self, *args, **kwargs),
    ),
    (
        'mean',
        lambda self, *args, **kwargs: getattr(np, 'mean')(
            self, *args, **kwargs),
    ),
    (
        'sum',
        lambda self, *args, **kwargs: getattr(np, 'sum')(
            self, *args, **kwargs),
    ),
):
    setattr(Variable, method, func)
    setattr(
        getattr(Variable, method), '__doc__',
        eval(f'np.ndarray.{method}').__doc__,
    )
