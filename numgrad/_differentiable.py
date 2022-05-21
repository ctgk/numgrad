from numgrad._config import config
from numgrad._graph import Graph
from numgrad._utils._isscalar import _isscalar
from numgrad._variable import Variable


def _func_to_grad(
    func,
    return_value: bool,
    force_scalar_output: bool,
) -> callable:

    def _grad_func(*args, **kwargs):
        if len(args) == 0:
            raise ValueError('Please pass at least one positional argument.')
        if config._graph is None:
            args = tuple(Variable(a) for a in args)
        with Graph(_allow_multiple_graphs=True) as g:
            value: Variable = func(*args, **kwargs)
        if force_scalar_output and (not _isscalar(value)):
            raise ValueError('Cannot compute gradient of non-scalar value.')
        grads = g.backward(value, args)
        if len(grads) == 1:
            grads = grads[0]
        if return_value:
            return (value._data, grads)
        return grads

    return _grad_func


def grad(forward_func: callable) -> callable:
    """Return a function that returns gradients of forward function.

    Parameters
    ----------
    forward_func : callable
        Input forward function. Note that the forward function must return
        scalar value.

    Returns
    -------
    callable
        Gradient function that returns gradients of the forward function with
        respect to given positional arguments.

    Examples
    --------
    >>> grad(np.tanh)(1)  # doctest: +ELLIPSIS
    0.4199...
    >>>
    >>> # returns tuple of gradients if you pass multiple positional args.
    >>> grad(np.hypot)(-3, 4)
    (-0.6, 0.8)
    >>>
    >>> # raises an error because `np.tanh([0, 1])` is not scalar.
    >>> grad(np.tanh)([0, 1])
    Traceback (most recent call last):
    ...
    ValueError: Cannot compute gradient of non-scalar value.
    """
    return _func_to_grad(
        forward_func, return_value=False, force_scalar_output=True)


def value_and_grad(forward_func: callable) -> callable:
    """Return a function that returns value and gradients of forward function.

    Parameters
    ----------
    forward_func : callable
        Input forward function. Note that the forward function must return
        scalar value.

    Returns
    -------
    callable
        Function that returns the resulting value of the forward function and
        its gradients with respect to given positional arguments.

    Examples
    --------
    >>> value_and_grad(np.tanh)(1)  # doctest: +ELLIPSIS
    (0.7615..., 0.4199...)
    >>>
    >>> # returns tuple of gradients if you pass multiple positional args.
    >>> value_and_grad(np.hypot)(-3, 4)
    (5.0, (-0.6, 0.8))
    >>>
    >>> # raises an error because `np.tanh([0, 1])` is not scalar.
    >>> value_and_grad(np.tanh)([0, 1])
    Traceback (most recent call last):
    ...
    ValueError: Cannot compute gradient of non-scalar value.
    """
    return _func_to_grad(
        forward_func, return_value=True, force_scalar_output=True)


def elementwise_grad(forward_func: callable) -> callable:
    """Return function that returns element-wise gradients of forward function.

    Parameters
    ----------
    forward_func : callable
        Input forward function. The return value does not have to be scalar
        unlike `grad`.

    Returns
    -------
    callable
        Function that returns element-wise gradient for given positional args.

    Examples
    --------
    >>> # this does not raise an error unlike `grad`.
    >>> elementwise_grad(np.tanh)([0, 1])
    array([1.        , 0.41997434])
    >>>
    >>> # returns tuple of gradients if you pass multiple positional args.
    >>> elementwise_grad(np.hypot)([-3, 3], 4)
    (array([-0.6,  0.6]), 1.6)
    """
    return _func_to_grad(
        forward_func, return_value=False, force_scalar_output=False)
