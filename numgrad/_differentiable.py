import inspect
import itertools
import typing as tp

from numgrad._graph import Graph
from numgrad._variable import Variable


class Differentiable:
    """Differentiable function.

    Examples
    --------
    >>> def tanh(a: 'Variable'):
    ...     a = np.exp(-2 * a)
    ...     return (1 - a) / (1 + a)
    ...
    >>> f = Differentiable(tanh)
    >>> f.grad(1)
    0.419974341614026
    >>> f.grad(a=1)
    {'a': 0.419974341614026}
    >>> f.value_and_grad(1)
    (0.7615941559557649, 0.419974341614026)
    >>> f.value_and_grad(a=1)
    (0.7615941559557649, {'a': 0.419974341614026})
    """

    def __init__(
        self,
        function: callable,
        variable_names: tp.Optional[tp.Union[str, tp.Tuple[str]]] = None,
    ) -> None:
        """Construct a differentiable function.

        Parameters
        ----------
        function : callable
            Original function.
        variable_names : tp.Optional[tp.Union[str, tp.Tuple[str]]], optional
            Name(s) of variable(s) to compute gradients for.
            By default None which search for parameters of the original
            function with type hint annotation of `'Variable'`.
        """
        self._function = function
        self._fullargspec = inspect.getfullargspec(function)
        if variable_names is None:
            self._variable_names = self._get_parameters_with_annotation_of(
                self._fullargspec, 'Variable')
        elif isinstance(variable_names, str):
            self._variable_names = (variable_names,)
        elif isinstance(variable_names, tuple):
            self._variable_names = variable_names
        else:
            raise TypeError(
                f'Invalid type for `variable_names`, {type(variable_names)}')

    def __call__(self, *args, **kwargs):
        """Call original function."""
        return self._function(*args, **kwargs)

    def value_and_grad(self, *args, **kwargs):
        """Get return value of the original function and gradients."""
        args, kwargs = self._preprocess_arguments(*args, **kwargs)
        with Graph() as g:
            value = self._function(*args, **kwargs)
        variables = [
            a for a in itertools.chain(args, kwargs.values())
            if isinstance(a, Variable)
        ]
        variable_ids = tuple(id(v) for v in variables)
        grads = g.gradient(value, variables)
        dargs = tuple(
            grads[variable_ids.index(id(a))] for a in args
            if id(a) in variable_ids
        )
        dkwargs = {
            k: grads[variables.index(a)]
            for k, a in kwargs.items() if a in variables
        }
        return (
            value._data,
            self._postprocess_grads(*dargs, **dkwargs),
        )

    def grad(self, *args, **kwargs):
        """Get gradients with respect to inputs."""
        return self.value_and_grad(*args, **kwargs)[1]

    @staticmethod
    def _get_parameters_with_annotation_of(
        fullargspec, annotation,
    ) -> tp.Tuple[str, ...]:
        return tuple(
            a for a in itertools.chain(
                fullargspec.args,
                (fullargspec.varargs,),
                fullargspec.kwonlyargs,
                (fullargspec.varkw,),
            )
            if fullargspec.annotations.get(a, None) == annotation
        )

    def _is_variable_name(self, name: str) -> bool:
        if name in self._variable_names:
            return True
        if name in self._fullargspec.kwonlyargs:
            return False
        return self._fullargspec.varkw in self._variable_names

    def _preprocess_arguments(self, *args, **kwargs):
        if len(args) <= len(self._fullargspec.args):
            args = tuple(
                Variable(a) if name in self._variable_names else a
                for a, name in zip(args, self._fullargspec.args)
            )
        else:
            args = tuple(
                Variable(a) if name in self._variable_names else a
                for a, name in
                itertools.zip_longest(
                    args,
                    self._fullargspec.args,
                    fillvalue=self._fullargspec.varargs,
                )
            )
        kwargs = {
            k: Variable(v) if self._is_variable_name(k) else v
            for k, v in kwargs.items()
        }
        return args, kwargs

    @staticmethod
    def _postprocess_dargs(*dargs):
        if len(dargs) == 0:
            return None
        if len(dargs) == 1:
            return dargs[0]
        return dargs

    @staticmethod
    def _postprocess_dkwargs(**dkwargs):
        if len(dkwargs) == 0:
            return None
        return dkwargs

    @classmethod
    def _postprocess_grads(cls, *dargs, **dkwargs):
        dargs = cls._postprocess_dargs(*dargs)
        dkwargs = cls._postprocess_dkwargs(**dkwargs)
        if dargs is None:
            return dkwargs
        if dkwargs is None:
            return dargs
        return dargs, dkwargs


def grad(
    function: callable,
    variable_names: tp.Optional[tp.Union[str, tp.Tuple[str]]] = None,
) -> callable:
    """Return gradient function of the given function.

    Parameters
    ----------
    function : callable
        Function to compute gradient of.
    variable_names : tp.Optional[tp.Union[str, tp.Tuple[str]]], optional
        Name(s) of variable(s) to compute gradient with respect to.
        By default None which computes gradient with respect to parameter with
        `'Variable'` annotation.

    Returns
    -------
    callable
        Gradient function that returns gradients with respect to variables.

    Examples
    --------
    >>> def tanh(a: 'Variable'):
    ...     a = np.exp(-2 * a)
    ...     return (1 - a) / (1 + a)
    ...
    >>> tanh_grad = grad(tanh)
    >>> tanh_grad(1)
    0.419974341614026
    >>> tanh_grad(a=1)
    {'a': 0.419974341614026}
    >>>
    >>> def hypot(a, b):  # note that there is no type hint
    ...     return np.sqrt(a * a + b * b)
    ...
    >>> grad(hypot, 'a')([-3, 3], 4)  # returns gradient wrt `a`
    array([-0.6,  0.6])
    >>> grad(hypot, ('a', 'b'))([-3, 3], 4)  # returns gradient wrt `a` and `b`
    (array([-0.6,  0.6]), 1.6)
    """
    return Differentiable(function, variable_names=variable_names).grad


def value_and_grad(
    function: callable,
    variable_names: tp.Optional[tp.Union[str, tp.Tuple[str]]] = None,
) -> callable:
    """Return function that returns the resulting value and gradients.

    Parameters
    ----------
    function : callable
        Function to compute its resulting value and gradients of.
    variable_names : tp.Optional[tp.Union[str, tp.Tuple[str]]], optional
        Name(s) of variable(s) to compute gradient with respect to.
        By default None which computes gradient with respect to parameter with
        `'Variable'` annotation.

    Returns
    -------
    callable
        Function that raeturns the resulting value of the original function and
        its gradients with respect to variables.

    Examples
    --------
    >>> def tanh(a: 'Variable'):
    ...     a = np.exp(-2 * a)
    ...     return (1 - a) / (1 + a)
    ...
    >>> tanh_value_and_grad = value_and_grad(tanh)
    >>> tanh_value_and_grad(1)
    (0.7615941559557649, 0.419974341614026)
    >>> tanh_value_and_grad(a=1)
    (0.7615941559557649, {'a': 0.419974341614026})
    >>>
    >>> def hypot(a, b):  # note that there is no type hint
    ...     return np.sqrt(a * a + b * b)
    ...
    >>> value_and_grad(hypot, 'a')([-3, 3], 4)
    (array([5., 5.]), array([-0.6,  0.6]))
    >>> value_and_grad(hypot, ('a', 'b'))([-3, 3], 4)
    (array([5., 5.]), (array([-0.6,  0.6]), 1.6))
    """
    return Differentiable(
        function, variable_names=variable_names).value_and_grad
