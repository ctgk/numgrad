import typing as tp

import numpy.typing as npt

from numgrad._graph import Graph
from numgrad._variable import Variable


class Differentiable:
    """Differentiable function.

    Examples
    --------
    >>> def tanh(a):
    ...     a = np.exp(-2 * a)
    ...     return (1 - a) / (1 + a)
    ...
    >>> f = Differentiable(tanh)
    >>> f.grad(1)
    0.419974341614026
    >>> f.grad(a=1)
    Traceback (most recent call last):
    ...
    ValueError: Please pass at least one positional argument.
    >>> f.value_and_grad(1)
    (0.7615941559557649, 0.419974341614026)
    >>> f.value_and_grad(a=1)
    Traceback (most recent call last):
    ...
    ValueError: Please pass at least one positional argument.
    """

    def __init__(self, function: callable) -> None:
        """Construct a differentiable function.

        Parameters
        ----------
        function : callable
            Original function.
        """
        if not callable(function):
            raise TypeError('`function` must be callable')
        self._function = function

    def __call__(self, *args, **kwargs):
        """Call original function."""
        return self._function(*args, **kwargs)

    def value_and_grad(
        self,
        *args: npt.ArrayLike,
        **kwargs,
    ) -> tp.Tuple[
        npt.ArrayLike, tp.Union[npt.ArrayLike, tp.Tuple[npt.ArrayLike, ...]],
    ]:
        """Return resulting value of the original function and gradients.

        Note that this function computes gradient with respect to all
        positional arguments.

        Returns
        -------
        Tuple[ArrayLike, Union[ArrayLike, Tuple[ArrayLike, ...]]]:
            Tuple of resulting value of the original function and gradient(s)
            with respect to positional argument(s).
        """
        if len(args) == 0:
            raise ValueError('Please pass at least one positional argument.')
        args = tuple(Variable(a) for a in args)
        with Graph() as g:
            value = self._function(*args, **kwargs)
        grads = g.gradient(value, args)
        return (value._data, (grads[0] if len(grads) == 1 else grads))

    def grad(
        self, *args: npt.ArrayLike, **kwargs,
    ) -> tp.Union[npt.ArrayLike, tp.Tuple[npt.ArrayLike, ...]]:
        """Return gradients of the original function with respect to args.

        Returns
        -------
        tp.Union[npt.ArrayLike, tp.Tuple[npt.ArrayLike, ...]]
            Gradient(s) of the original function with respect to positional
            argument(s)
        """
        return self.value_and_grad(*args, **kwargs)[1]


def grad(function: callable) -> callable:
    """Return gradient function of the given function.

    Parameters
    ----------
    function : callable
        Function to compute gradient of.

    Returns
    -------
    callable
        Gradient function that returns gradients with respect to positional
        arguments.

    Examples
    --------
    >>> def tanh(a):
    ...     a = np.exp(-2 * a)
    ...     return (1 - a) / (1 + a)
    ...
    >>> tanh_grad = grad(tanh)
    >>> tanh_grad(1)
    0.419974341614026
    >>> tanh_grad(a=1)
    Traceback (most recent call last):
    ...
    ValueError: Please pass at least one positional argument.
    >>>
    >>> def hypot(a, b):
    ...     return np.sqrt(a * a + b * b)
    ...
    >>> grad(hypot)([-3, 3], b=4)  # returns gradient wrt `a`
    array([-0.6,  0.6])
    >>> grad(hypot)([-3, 3], 4)    # returns gradient wrt `a` and `b`
    (array([-0.6,  0.6]), 1.6)
    """
    return Differentiable(function).grad


def value_and_grad(function: callable) -> callable:
    """Return function that returns the resulting value and gradients.

    Parameters
    ----------
    function : callable
        Function to compute its resulting value and gradients of.

    Returns
    -------
    callable
        Function that raeturns the resulting value of the original function and
        its gradients with respect to positional arguments.

    Examples
    --------
    >>> def tanh(a):
    ...     a = np.exp(-2 * a)
    ...     return (1 - a) / (1 + a)
    ...
    >>> tanh_value_and_grad = value_and_grad(tanh)
    >>> tanh_value_and_grad(1)
    (0.7615941559557649, 0.419974341614026)
    >>> tanh_value_and_grad(a=1)
    Traceback (most recent call last):
    ...
    ValueError: Please pass at least one positional argument.
    >>>
    >>> def hypot(a, b):
    ...     return np.sqrt(a * a + b * b)
    ...
    >>> value_and_grad(hypot)([-3, 3], b=4)
    (array([5., 5.]), array([-0.6,  0.6]))
    >>> value_and_grad(hypot)([-3, 3], 4)
    (array([5., 5.]), (array([-0.6,  0.6]), 1.6))
    """
    return Differentiable(function).value_and_grad
