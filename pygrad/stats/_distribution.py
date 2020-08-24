import abc
import inspect
import typing as tp

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._utils._typecheck import _typecheck


class Distribution(Module, abc.ABC):
    """Base probability distribution class
    """

    def __init__(
            self,
            rv: str = 'x',
            name: str = 'p'):
        super().__init__()
        self._rv = (self._ensure_no_ng_char('rv', rv),)
        conditions = inspect.getfullargspec(self.forward).args
        try:
            conditions.remove('self')
        except ValueError:
            pass
        if self._rv[0] in conditions:
            raise ValueError()
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
        out = f'{self._name}({self._rv[0]}'
        if self._conditions:
            out += '|' + ','.join(self._conditions)
        out += ')'
        return out

    def __call__(
            self,
            *,
            use_cache: bool = False,
            **conditions: Array) -> tp.Dict[str, Array]:
        """Return statistics of this distribution given conditions

        Be sure to call clear() method to compute statistics again,
        otherwise the cache value will be returned.
        """
        if self._stats is None or (not use_cache):
            self._stats = self.forward(**conditions)
        return self._stats

    def clear(self):
        super().clear()
        self._stats = None

    @_typecheck(exclude_types=(Array,))
    def logpdf(
            self,
            obs: tp.Union[Array, tp.Dict[str, Array]],
            conditions: tp.Dict[str, Array] = {},
            *,
            use_cache: bool = False) -> Array:
        """Return sum of logarithm of probability density (mass) function
        given conditions

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
        return self._logpdf(
            obs[self._rv[0]] if isinstance(obs, dict) else obs,
            **statistics).sum()

    @_typecheck(exclude_types=(Array,))
    def sample(
            self,
            conditions: tp.Dict[str, Array] = {},
            *,
            use_cache: bool = False) -> tp.Dict[str, Array]:
        """Return random sample of the random variable given conditions

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
        return {self._rv[0]: self._sample(**statistics)}

    @abc.abstractmethod
    def forward(self, **conditions: Array) -> tp.Dict[str, Array]:
        """Return statistics of this distribution given conditions
        """
        pass

    @abc.abstractmethod
    def _logpdf(self) -> Array:
        pass

    @abc.abstractmethod
    def _sample(self) -> Array:
        pass


class JointDistribution(Distribution):
    """Joint probability distribution class
    """

    def __init__(self, p1: Distribution, p2: Distribution):
        Module.__init__(self)
        if (set(p1._rv) & set(p2._rv)) or (set(p2._conditions) & set(p1._rv)):
            raise ValueError(f'Invalid joint of distributions {p1} and {p2}')
        self.p1 = p1
        self.p2 = p2
        self._rv = tuple(set(p1._rv) | set(p2._rv))
        self._conditions = tuple(
            (set(p1._conditions) | set(p2._conditions)) - set(self._rv))

    def __repr__(self):
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
            use_cache: bool = False) -> Array:
        """Return sum of logarithm of probability density (mass) function
        given conditions

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
            use_cache=use_cache)
        p2_logpdf = self.p2.logpdf(
            obs=self._get_rv_of(self.p2, **obs, **conditions),
            conditions=self._get_conditions_of(self.p2, **obs, **conditions),
            use_cache=use_cache)
        return p1_logpdf + p2_logpdf

    @_typecheck(exclude_types=(Array,))
    def sample(
            self,
            conditions: tp.Dict[str, Array] = {},
            *,
            use_cache: bool = False) -> tp.Dict[str, Array]:
        """Return random sample of the random variable given conditions

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
        raise NotImplementedError

    def _logpdf(self) -> Array:
        raise NotImplementedError

    def _sample(self) -> Array:
        raise NotImplementedError
