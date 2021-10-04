import abc
import inspect
import typing as tp

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._utils._typecheck import _typecheck
from pygrad.stats._statistics import Statistics


class Distribution(Module, abc.ABC):
    """Base probability distribution class."""

    def __init__(
        self,
        rv: str = 'x',
        name: str = 'p',
        conditions: tp.List[str] = None,
    ):
        """Initialize probability distribution module.

        Parameters
        ----------
        rv : str, optional
            Name of the random variable that follows the distribution,
            by default 'x'
        name : str, optional
            Name of the distribution, by default 'p'
        conditions : tp.List[str], optional
            Name of conditional random variables, by default None
        """
        super().__init__()
        self._rv = (self._ensure_no_ng_char('rv', rv),)
        forward_spec = inspect.getfullargspec(self.forward)
        forward_args = forward_spec.args
        try:
            forward_args.remove('self')
        except ValueError:
            pass
        if conditions is None:
            conditions = forward_args
        else:
            if len(conditions) != len(forward_args):
                raise ValueError(
                    f'{len(conditions)}(len(conditions)) != '
                    f'{len(forward_args)}(# of forward() args)')
        if self._rv[0] in conditions:
            raise ValueError()
        if 'return' in forward_spec.annotations:
            return_annotation = forward_spec.annotations['return']
            if not issubclass(return_annotation, Statistics):
                raise TypeError(
                    'forward() method should return instance of Statistics')
        else:
            raise ValueError('forward() method should have return annotation')
        self._conditions = tuple(conditions)
        self._name = self._ensure_no_ng_char('name', name)
        self._stats = None

    def _ensure_no_ng_char(self, name: str, value: str):
        for ng_char in (',', '(', ')', '|'):
            if ng_char in value:
                raise ValueError(
                    f'NG character {ng_char} in arg \'{name}\', {value}')
        return value

    def __repr__(self) -> str:
        """Return representation of the distribution.

        Returns
        -------
        str
            Representation of the distribution.
        """
        out = f'{self._name}({self._rv[0]}'
        if self._conditions:
            out += '|' + ','.join(self._conditions)
        out += ')'
        return out

    def __call__(
        self,
        *,
        use_cache: bool = False,
        **conditions: Array,
    ) -> Statistics:
        """Return statistics of the distribution given conditions.

        Parameters
        ----------
        use_cache : bool, optional
            Use cached values if True, by default False

        Returns
        -------
        Statistics
            Statistics of the distribution.
        """
        if self._stats is None or (not use_cache):
            self._stats = self.forward(
                *tuple(conditions[c] for c in self._conditions))
        return self._stats

    def clear(self):
        """Clear caches."""
        super().clear()
        self._stats = None

    @_typecheck(exclude_types=(Array,))
    def logpdf(
        self,
        obs: tp.Union[Array, tp.Dict[str, Array]],
        conditions: tp.Dict[str, Array] = {},
        *,
        use_cache: bool = False,
        reduce: tp.Union[str, None] = 'sum',
    ) -> Array:
        """Return sum of logarithm of pdf (or pmf) given conditions.

        Parameters
        ----------
        obs : tp.Union[Array, tp.Dict[str, Array]]
            Observation of the random variable
        conditions : tp.Dict[str, Array], optional
            Dict of conditional variables, by default {}
        use_cache : bool, optional
            Use cached statistics if True and available, by default False

        Returns
        -------
        Array
            Summation of logarithm of probability density (mass) function
        """
        statistics = self.__call__(use_cache=use_cache, **conditions)
        out = statistics.logpdf(
            obs[self._rv[0]] if isinstance(obs, dict) else obs)
        if reduce == 'sum':
            return out.sum()
        elif reduce == 'mean':
            return out.mean()
        else:
            return out

    @_typecheck(exclude_types=(Array,))
    def sample(
        self,
        conditions: tp.Dict[str, Array] = {},
        *,
        use_cache: bool = False,
    ) -> tp.Dict[str, Array]:
        """Return random sample of the random variable given conditions.

        Parameters
        ----------
        conditions : tp.Dict[str, Array], optional
            Dict of conditional variables, by default {}
        use_cache : bool, optional
            Use cached statistics if True and available, by default False

        Returns
        -------
        Dict[str, Array]
            Random sample of the random variable
        """
        statistics = self.__call__(use_cache=use_cache, **conditions)
        return {self._rv[0]: statistics.sample()}

    @abc.abstractmethod
    def forward(self, **conditions: Array) -> Statistics:
        """Return statistics of this distribution given conditions."""
        pass


class JointDistribution(Distribution):
    """Joint probability distribution class."""

    def __init__(self, p1: Distribution, p2: Distribution):
        """Initialize joint distribution object.

        Parameters
        ----------
        p1 : Distribution
            First argument of distribution.
        p2 : Distribution
            Second argument of distribution.
        """
        Module.__init__(self)
        if (set(p1._rv) & set(p2._rv)) or (set(p2._conditions) & set(p1._rv)):
            raise ValueError(f'Invalid joint of distributions {p1} and {p2}')
        self.p1 = p1
        self.p2 = p2
        self._rv = tuple(set(p1._rv) | set(p2._rv))
        self._conditions = tuple(
            (set(p1._conditions) | set(p2._conditions)) - set(self._rv))

    def __repr__(self):
        """Return representation of the joint distribution.

        Returns
        -------
        str
            Representation of the joint distribution.
        """
        return repr(self.p1) + repr(self.p2)

    def _get_rv_of(self, p: Distribution, **kwargs):
        return {k: v for k, v in kwargs.items() if k in p._rv}

    def _get_conditions_of(self, p: Distribution, **kwargs):
        return {k: v for k, v in kwargs.items() if k in p._conditions}

    @_typecheck(exclude_types=(Array,))
    def logpdf(
        self,
        obs: tp.Dict[str, Array],
        conditions: tp.Dict[str, Array] = {},
        *,
        use_cache: bool = False,
        reduce: tp.Union[str, None] = 'sum',
    ) -> Array:
        """Return sum of logarithm of pdf (pmf) given conditions.

        Parameters
        ----------
        obs : tp.Union[Array, tp.Dict[str, Array]]
            Observation of the random variable
        conditions : tp.Dict[str, Array], optional
            Dict of conditional variables, by default {}
        use_cache : bool, optional
            Use cached statistics if True and available, by default False

        Returns
        -------
        Array
            Summation of logarithm of probability density (mass) function
        """
        p1_logpdf = self.p1.logpdf(
            obs=self._get_rv_of(self.p1, **obs, **conditions),
            conditions=self._get_conditions_of(self.p1, **obs, **conditions),
            use_cache=use_cache, reduce=reduce)
        p2_logpdf = self.p2.logpdf(
            obs=self._get_rv_of(self.p2, **obs, **conditions),
            conditions=self._get_conditions_of(self.p2, **obs, **conditions),
            use_cache=use_cache, reduce=reduce)
        return p1_logpdf + p2_logpdf

    @_typecheck(exclude_types=(Array,))
    def sample(
        self,
        conditions: tp.Dict[str, Array] = {},
        *,
        use_cache: bool = False,
    ) -> tp.Dict[str, Array]:
        """Return random sample of the random variable given conditions.

        Parameters
        ----------
        conditions : tp.Dict[str, Array], optional
            Dict of conditional variables, by default {}
        use_cache : bool, optional
            Use cached statistics if True and available, by default False

        Returns
        -------
        Dict[str, Array]
            Random sample of the random variable
        """
        p2_sample = self.p2.sample(
            self._get_conditions_of(self.p2, **conditions))
        p1_sample = self.p1.sample(
            self._get_conditions_of(self.p1, **conditions, **p2_sample))
        return {**p1_sample, **p2_sample}

    def forward(self, **conditions: Array):
        """Return statistics."""
        raise NotImplementedError

    def _logpdf(self) -> Array:
        raise NotImplementedError

    def _sample(self) -> Array:
        raise NotImplementedError
