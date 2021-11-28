"""Module for random operations."""

from pygrad.random._exponential import exponential
from pygrad.random._gumbel_sigmoid import gumbel_sigmoid
from pygrad.random._gumbel_softmax import gumbel_softmax
from pygrad.random._normal import normal
from pygrad.random._uniform import uniform


_functions = [
    exponential,
    gumbel_sigmoid,
    gumbel_softmax,
    normal,
    uniform,
]


__all__ = [_func.__name__ for _func in _functions]
