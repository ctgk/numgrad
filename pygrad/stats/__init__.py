"""Statistics module."""

from pygrad.stats._log_softmax import log_softmax
from pygrad.stats._sigmoid import sigmoid
from pygrad.stats._sigmoid_cross_entropy import sigmoid_cross_entropy
from pygrad.stats._softmax import softmax
from pygrad.stats._softmax_cross_entropy import softmax_cross_entropy
from pygrad.stats._sparse_softmax_cross_entropy import (
    sparse_softmax_cross_entropy,
)


_functions = [
    log_softmax,
    sigmoid,
    sigmoid_cross_entropy,
    softmax,
    softmax_cross_entropy,
    sparse_softmax_cross_entropy,
]


__all__ = [_func.__name__ for _func in _functions]
