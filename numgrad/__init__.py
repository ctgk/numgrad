"""Simple gradient computation library for Python."""

from numgrad._config import Config, config  # noqa: F401
from numgrad._decorators import differentiable
from numgrad._differentiable import Differentiable, grad, value_and_grad
from numgrad._graph import Graph
from numgrad._variable import Variable

from numgrad import _numpy, _scipy  # noqa: F401, I100, I202

from numgrad._version import __version__  # noqa: F401, I202


_classes = [
    Config,
    Differentiable,
    Graph,
    Variable,
]


_functions = [
    differentiable,
    grad,
    value_and_grad,
]


for _cls in _classes:
    _cls.__module__ = __name__


__all__ = (
    ['config']
    + [_cls.__name__ for _cls in _classes]
    + [_f.__name__ for _f in _functions]
)


del _cls
del _classes
