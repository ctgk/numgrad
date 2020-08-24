from pygrad.stats._bernoulli import Bernoulli
from pygrad.stats._categorical import Categorical
from pygrad.stats._distribution import Distribution, JointDistribution
from pygrad.stats._log_softmax import log_softmax
from pygrad.stats._normal import Normal
from pygrad.stats._relaxed_bernoulli import RelaxedBernoulli
from pygrad.stats._relaxed_categorical import RelaxedCategorical
from pygrad.stats._sigmoid import sigmoid
from pygrad.stats._sigmoid_cross_entropy import sigmoid_cross_entropy
from pygrad.stats._softmax import softmax
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy
from pygrad.stats._sparse_softmax_cross_entropy import (
    sparse_softmax_cross_entropy,
)


_classes = [
    Bernoulli,
    Categorical,
    Distribution,
    JointDistribution,
    Normal,
    RelaxedBernoulli,
    RelaxedCategorical,
]


for _cls in _classes:
    _cls.__module__ = 'pygrad.stats'


Distribution.__mul__ = lambda p1, p2: JointDistribution(p1, p2)


__all__ = [_cls.__name__ for _cls in _classes] + [
    'log_softmax',
    'sigmoid',
    'sigmoid_cross_entropy',
    'softmax',
    'softmax_cross_entropy',
    'sparse_softmax_cross_entropy',
]
