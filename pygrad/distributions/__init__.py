"""Distribution module."""

from pygrad.distributions._bernoulli import Bernoulli
from pygrad.distributions._categorical import Categorical
from pygrad.distributions._distribution import Distribution, JointDistribution
from pygrad.distributions._exponential import Exponential
from pygrad.distributions._normal import Normal
from pygrad.distributions._relaxed_bernoulli import RelaxedBernoulli
from pygrad.distributions._relaxed_categorical import RelaxedCategorical


_classes = [
    Bernoulli,
    Categorical,
    Distribution,
    Exponential,
    JointDistribution,
    Normal,
    RelaxedBernoulli,
    RelaxedCategorical,
]


for _cls in _classes:
    _cls.__module__ = 'pygrad.distributions'


Distribution.__mul__ = lambda a, b: JointDistribution(a, b)


__all__ = [_cls.__name__ for _cls in _classes]
