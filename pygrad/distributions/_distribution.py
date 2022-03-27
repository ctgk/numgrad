import abc
import inspect
import typing as tp

from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._math._sum import sum
from pygrad._utils._typecheck import _typecheck


class BaseDistribution(Module):
    """Base distribution that a random variable follow.

    .. math::
        p(x|...)
    """

    def __init__(self, notation: str) -> None:
        """Initialize base distribution class.

        Parameters
        ----------
        notation : str
            Notation of the distribution.
        """
        super().__init__()
        self._check_notation_and_set_properties(notation=notation)

    def __repr__(self) -> str:
        """Return representation of the distribution.

        Returns
        -------
        str
            Representation of the distribution.
        """
        notation = self._distribution_name + '(' + ','.join(
            self._random_variable_names)
        if len(self._conditional_variable_names) != 0:
            notation += '|' + ','.join(self._conditional_variable_names)
        notation += ')'
        return notation

    @abc.abstractmethod
    def logp(
        self,
        observed: tp.Dict[str, TensorLike],
        conditions: tp.Dict[str, TensorLike],
        **kwargs,
    ) -> Tensor:
        """Return logarithm of probability density (mass) function.

        Parameters
        ----------
        observed : tp.Dict[str, TensorLike]
            Observed value(s) of random variable(s)
        conditions : tp.Dict[str, TensorLike]
            Value(s) of conditional random variable(s)

        Returns
        -------
        Tensor
            Logarithm of probability density (mass) function.
        """
        pass

    @abc.abstractmethod
    def sample(
        self,
        conditions: tp.Dict[str, TensorLike],
        **kwargs,
    ) -> tp.Union[Tensor, tp.Dict[str, Tensor]]:
        """Return random sample from the distribution.

        Parameters
        ----------
        conditions : tp.Dict[str, TensorLike]
            Value(s) of conditional random variable(s)

        Returns
        -------
        tp.Union[Tensor, tp.Dict[str, Tensor]]
            Random sample from the distribution.
        """
        pass

    def __mul__(self, other: 'BaseDistribution') -> 'JointDistribution':
        """Return joint probability of the random variables.

        Parameters
        ----------
        other : BaseDistribution
            Distribution that another random variable(s) follow

        Returns
        -------
        JointDistribution
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        if not isinstance(other, BaseDistribution):
            raise ValueError(
                'Multiplying `BaseDistribution` and '
                f'{type(other)} is not supported')
        return JointDistribution(self, other)

    def _check_notation_and_set_properties(self, notation: str):
        if (
            notation.count('(') != 1
            or notation.count(')') != 1
            or notation.count('|') >= 2
        ):
            raise ValueError(f'Invalid notation, {notation}')
        notation = ''.join(notation.split())  # remove all kinds of
        self._distribution_name, remaining = notation.split('(')
        if self._distribution_name == '':
            raise ValueError(f'Empty distribution name, {notation}')
        remaining = remaining.rstrip(')')
        if '|' in remaining:
            remaining, conditional_str = remaining.split('|')
            self._conditional_variable_names = conditional_str.split(',')
            if any(n == '' for n in self._conditional_variable_names):
                raise ValueError(
                    f'Empty conditional variable name, {notation}')
        else:
            self._conditional_variable_names = []
        self._random_variable_names = remaining.split(',')
        if any(n == '' for n in self._random_variable_names):
            raise ValueError(f'Empty random variable name, {notation}')
        if any(
            rv in self._conditional_variable_names
            for rv in self._random_variable_names
        ):
            raise ValueError(
                'A distribution of a random variable may not be conditioned '
                f'by the random variable itself. {notation}.',
            )


class Distribution(BaseDistribution):
    """Probability distribution that a random variable follows.

    Use this class to create your custom probability distribution.

    Examples
    --------
    >>> class ConditionedNormal(gd.distributions.Distribution):
    ...
    ...     def __init__(self, notation: str) -> None:
    ...         super().__init__(notation)
    ...
    ...     def forward(self, mu) -> gd.distributions.Normal:
    ...         # 1. Parameters must contain conditional variable(s).
    ...         # 2. Return type hint must be subclass of `BasicDistribution`.
    ...         return gd.distributions.Normal(loc=mu, scale=1.)
    ...
    >>> # conditional variable(s) may have suffix.
    >>> px = ConditionedNormal('p(y|mu_0)')
    >>> px
    p(y|mu_0)
    >>> px.logp(observed=0, conditions={'mu_0': 1})
    Tensor(-1.41893853)
    >>> px.sample(conditions={'mu_0': 0})  # doctest: +SKIP
    {'y': Tensor(-1.30001032)}
    """

    def __init__(self, notation: str = 'p(x)') -> None:
        """Initialize a distribution.

        Parameters
        ----------
        notation : str, optional
            Notation of the distribution, by default 'p(x)'
        """
        super().__init__(notation)
        if len(self._random_variable_names) != 1:
            raise ValueError(
                'Non-joint probability distribution may have '
                f'only one random variable, but {notation} contains multiple.',
            )
        self._check_forward()

    def __call__(
        self,
        conditions: tp.Dict[str, TensorLike] = {},
        **kwargs,
    ) -> 'BasicDistribution':
        """Return basic distributions given conditions.

        Parameters
        ----------
        conditions : tp.Dict[str, TensorLike], optional
            Conditional random variables, by default {}

        Returns
        -------
        BasicDistribution
            Basic distribution of the random variable.

        Raises
        ------
        ValueError
            Given conditions does not match required conditions.
        """
        if set(conditions) != set(self._conditional_variable_names):
            raise ValueError(
                f'Given conditions {tuple(set(conditions))} does not match '
                'required conditions '
                f'{tuple(self._conditional_variable_names)}.')
        p = self.forward(**self._remove_suffix(**conditions), **kwargs)
        p._distribution_name = self._distribution_name
        p._random_variable_names[0] = self._random_variable_names[0]
        return p

    @abc.abstractmethod
    def forward(self) -> 'BasicDistribution':  # noqa: D102
        pass

    def entropy(self) -> Tensor:
        r"""Return (differential) entropy of the random variable.

        .. math::
            H[x] = \begin{cases}
                -\sum_x p(x)\ln p(x) & ({\rm discrete}) \\
                -\int p(x)\ln p(x){\rm d}x & ({\rm continuous})
            \end{cases}

        Returns
        -------
        Tensor
            Entropy (or differential entropy) of the random variable.
        """
        if len(self._conditional_variable_names) == 0:
            return self.__call__()._entropy()
        raise NotImplementedError('Conditional entropy not supported yet.')

    @_typecheck()
    def logp(
        self,
        observed: tp.Union[TensorLike, tp.Dict[str, TensorLike]],
        conditions: tp.Dict[str, TensorLike] = {},
        **kwargs,
    ) -> Tensor:
        r"""Return logarithm of probability mass (or density) function.

        .. math::
            \ln p(x)

        Parameters
        ----------
        observed : tp.Union[TensorLike, tp.Dict[str, TensorLike]]
            Observed value of the random variable.
        conditions : tp.Dict[str, TensorLike], optional
            Values of conditional variables, by default {}

        Returns
        -------
        Tensor
            Logarithm of probability mass (or density) function.
        """
        if isinstance(observed, dict):
            assert set(observed) == set(self._random_variable_names)
            observed = observed[self._random_variable_names[0]]
        return self.__call__(conditions, **kwargs)._logp(observed)

    @_typecheck()
    def sample(
        self,
        conditions: tp.Dict[str, TensorLike] = {},
        return_tensor: bool = False,
        **kwargs,
    ) -> tp.Union[Tensor, tp.Dict[str, Tensor]]:
        r"""Return random sample of the random variable.

        .. math::
            x_i \sim p(x)

        Parameters
        ----------
        conditions : tp.Dict[str, TensorLike], optional
            Values of conditional variables, by default {}
        return_tensor : bool, optional
            Return tensor object if true else return dict object,
            by default False

        Returns
        -------
        tp.Union[Tensor, tp.Dict[str, Tensor]]
            Random sample of the random variable.
        """
        p = self.__call__(conditions, **kwargs)
        if return_tensor:
            return p._sample()
        return {self._random_variable_names[0]: p._sample()}

    def _remove_suffix(self, **conditions) -> dict:
        args = inspect.getfullargspec(self.__call__).args
        return {
            (k if k in args else k.split('_')[0]): v
            for k, v in conditions.items()
        }

    def _check_forward(self):
        fullspec = inspect.getfullargspec(self.forward)
        args = fullspec.args
        try:
            args.remove('self')
        except ValueError:
            pass
        if len(self._conditional_variable_names) != len(args):
            raise ValueError(
                f'{len(self._conditional_variable_names)}(# of conditions)'
                f'!={len(args)}(# of forward method parameters)',
            )
        if 'return' in fullspec.annotations:
            return_annotation = fullspec.annotations['return']
            if return_annotation == 'BasicDistribution':
                pass
            elif not issubclass(return_annotation, BasicDistribution):
                raise TypeError(
                    '`forward()` method must return instance of '
                    f'`BasicDistribution`, not {return_annotation}',
                )
        else:
            raise ValueError('`forward()` method must have return type hint.')


class BasicDistribution(Distribution):
    """Basic distribution commonly used for probabilistic models."""

    def __init__(self, notation: str) -> None:  # noqa: D107
        super().__init__(notation)
        if len(self._conditional_variable_names) > 0:
            raise ValueError(
                'Basic probability distribution may not be conditioned. '
                f'{notation}.',
            )

    def forward(self) -> 'BasicDistribution':  # noqa: D102
        return self

    @abc.abstractmethod
    def _entropy(self) -> Tensor:
        pass

    @abc.abstractmethod
    def _logp(self, observed: TensorLike) -> Tensor:
        pass

    @abc.abstractmethod
    def _sample(self) -> Tensor:
        pass


def _have_overlap(a: tp.Iterable, b: tp.Iterable) -> bool:
    return bool(set(a).intersection(set(b)))


def _get_notation(p1: BaseDistribution, p2: BaseDistribution) -> str:
    if p1._distribution_name == p2._distribution_name:
        name = p1._distribution_name
    else:
        name = 'p'
    rv = list(set(p1._random_variable_names + p2._random_variable_names))
    cond = list(
        set(p1._conditional_variable_names + p2._conditional_variable_names)
        - set(rv))
    notation = name + '(' + ','.join(rv)
    if len(cond) > 0:
        notation += '|' + ','.join(cond)
    notation += ')'
    return notation


class JointDistribution(BaseDistribution):
    """Distribution for joint random variables.

    .. math::
        p(x, y|...)
    """

    def __init__(self, p1: BaseDistribution, p2: BaseDistribution) -> None:
        """Construct joint distribution given two distributions.

        Parameters
        ----------
        p1 : BaseDistribution
            First distribution
        p2 : BaseDistribution
            Second distribution
        """
        super().__init__(_get_notation(p1, p2))
        if _have_overlap(p1._random_variable_names, p2._random_variable_names):
            raise ValueError(
                f'Invalid multiplication of distributions: {p1} * {p2}')
        if not _have_overlap(
            p1._random_variable_names,
            p2._conditional_variable_names,
        ):
            self._conditional = p1
            self._independent = p2
        elif not _have_overlap(
            p1._conditional_variable_names,
            p2._random_variable_names,
        ):
            self._conditional = p2
            self._independent = p1
        else:
            raise ValueError(
                f'Invalid multiplication of distributions: {p1} * {p2}')

    def __repr__(self) -> str:
        """Return representation of the distribution.

        Returns
        -------
        str
            Representation of the distribution.
        """
        return repr(self._conditional) + repr(self._independent)

    def __call__(self):  # noqa: D102
        raise NotImplementedError

    def _entropy_of(
        self,
        p: BaseDistribution,
        reduce: callable = sum,
    ) -> Tensor:
        if isinstance(p, Distribution):
            return reduce(p.entropy())
        return p.entropy(reduce=reduce)

    def entropy(self, *, reduce: callable = sum) -> Tensor:
        r"""Return (differential) entropy of the random variable.

        .. math::
            H[x] = \begin{cases}
                -\sum_x p(x)\ln p(x) & ({\rm discrete}) \\
                -\int p(x)\ln p(x){\rm d}x & ({\rm continuous})
            \end{cases}

        Parameters
        ----------
        reduce : callable, optional
            Callable to reduce axes of tensors.

        Returns
        -------
        Tensor
            Entropy (or differential entropy) of the random variable.
        """
        if len(self._conditional_variable_names) == 0:
            return (
                self._entropy_of(self._conditional, reduce)
                + self._entropy_of(self._independent, reduce)
            )
        raise NotImplementedError('Conditional entropy not supported yet.')

    @staticmethod
    def _logp_of(
        p: BaseDistribution,
        variables: dict,
        reduce: callable,
        **kwargs,
    ):
        observed = {
            k: v for k, v in variables.items()
            if k in p._random_variable_names
        }
        if isinstance(p, Distribution):
            assert set(observed) == set(p._random_variable_names)
            observed = observed[p._random_variable_names[0]]
            return reduce(p.logp(
                observed=observed,
                conditions={
                    k: v for k, v in variables.items()
                    if k in p._conditional_variable_names
                },
                **kwargs,
            ))
        return p.logp(
            observed=observed,
            conditions={
                k: v for k, v in variables.items()
                if k in p._conditional_variable_names
            },
            reduce=reduce,
            **kwargs,
        )

    @_typecheck()
    def logp(
        self,
        observed: tp.Dict[str, TensorLike],
        conditions: tp.Dict[str, TensorLike] = {},
        *,
        reduce: callable = sum,
        **kwargs,
    ) -> Tensor:
        r"""Return logarithm of probability mass (or density) function.

        .. math::
            \ln p(x, y)

        Parameters
        ----------
        observed : tp.Dict[str, TensorLike]
            Observed values of the random variables.
        conditions : tp.Dict[str, TensorLike], optional
            Values of conditional variables, by default {}
        reduce : callable
            Callable to reduce axes of tensors, by default sum

        Returns
        -------
        Tensor
            Logarithm of probability mass (or density) function.
        """
        assert set(observed) == set(self._random_variable_names)
        assert set(conditions) == set(self._conditional_variable_names)
        logp_inde = self._logp_of(
            self._independent, {**observed, **conditions}, reduce, **kwargs)
        logp_cond = self._logp_of(
            self._conditional, {**observed, **conditions}, reduce, **kwargs)
        return logp_cond + logp_inde

    @staticmethod
    def _sample_of(p: BaseDistribution, variables: dict, **kwargs) -> dict:
        out = p.sample(
            conditions={
                k: v for k, v in variables.items()
                if k in p._conditional_variable_names
            },
            **kwargs,
        )
        if not isinstance(out, dict):
            assert isinstance(p, Distribution)
            out = {p._random_variable_names[0]: out}
        return out

    @_typecheck()
    def sample(
        self,
        conditions: tp.Dict[str, TensorLike] = {},
        **kwargs,
    ) -> tp.Dict[str, Tensor]:
        r"""Return random sample of the random variables.

        .. math::
            x_i \sim p(x)

        Parameters
        ----------
        conditions : tp.Dict[str, TensorLike], optional
            Values of conditional variables, by default {}

        Returns
        -------
        tp.Dict[str, Tensor]
            Random sample of the random variables.
        """
        assert set(conditions) == set(self._conditional_variable_names)
        sample_inde = self._sample_of(self._independent, conditions, **kwargs)
        sample_cond = self._sample_of(
            self._conditional, {**conditions, **sample_inde}, **kwargs)
        return {**sample_inde, **sample_cond}
