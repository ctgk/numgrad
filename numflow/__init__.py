"""Simple gradient computation library for Python."""

from numflow._config import Config, config  # noqa: F401
from numflow._decorators import differentiable
from numflow._graph import Graph
from numflow._variable import Variable

from numflow import _numpy, _scipy  # noqa: F401, I100, I202

from numflow._version import __version__  # noqa: F401, I202


_classes = [
    Config,
    Graph,
    Variable,
]


_functions = [
    differentiable,
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
