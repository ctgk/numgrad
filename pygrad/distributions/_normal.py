import typing as tp

from pygrad._core._tensor import TensorLike
from pygrad._utils._typecheck import _typecheck
from pygrad.distributions._distribution import Distribution
from pygrad.stats._normal import Normal as NormalStats


class Normal(Distribution):
    """Normal distribution aka Gaussian distribution.

    Examples
    --------
    >>> np.random.seed(0)
    >>> n = gd.distributions.Normal(rv='x')
    >>> n
    N(x)
    >>> n.logpdf(1)
    Tensor(-1.41893853)
    >>> n.sample()['x']
    Tensor(1.76405235)
    >>> n.sample()['x']
    Tensor(0.40015721)
    """

    @_typecheck()
    def __init__(
        self,
        rv: str = 'x',
        name: str = 'N',
        *,
        conditions: tp.Union[tp.List[str], tp.Tuple[str, ...], None] = None,
        loc: TensorLike = 0,
        scale: TensorLike = 1,
    ):
        """Initialize normal distribution.

        Parameters
        ----------
        rv : str, optional
            Name of the random variable that follows the distribution,
            by default 'x'
        name : str, optional
            Name of the distribution, by default 'N'
        conditions : tp.Union[tp.List[str], tp.Tuple[str, ...], None], optional
            Names of the random variables if exist, by default None
        loc : TensorLike, optional
            Location parameter of the normal distribution, by default 0
        scale : TensorLike, optional
            Scale parameter of the normal distribution, by default 1
        """
        super().__init__(rv=rv, name=name, conditions=conditions)
        self._loc = loc
        self._scale = scale

    def forward(self) -> NormalStats:
        """Return statistics of the normal distribution.

        Returns
        -------
        NormalStats
            Statistics of the normal distribution.
        """
        return NormalStats(self._loc, self._scale)
