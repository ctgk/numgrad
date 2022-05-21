"""Simple gradient computation library for Python."""

from numgrad._config import Config, config  # noqa: F401
from numgrad._differentiable import elementwise_grad, grad, value_and_grad
from numgrad._graph import Graph
from numgrad._utils._has_vjp import has_vjp
from numgrad._variable import Variable
from numgrad._vjp import custom_vjp

from numgrad import _numpy, _scipy  # noqa: F401, I100, I202

from numgrad._version import __version__  # noqa: F401, I202


_classes = [
    Config,
    Graph,
    Variable,
]


_functions = [
    custom_vjp,
    elementwise_grad,
    grad,
    has_vjp,
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
