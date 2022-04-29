"""Simple gradient computation library for Python."""

from pygrad._config import Config, config  # noqa: F401
from pygrad._graph import Graph
from pygrad._variable import Variable

from pygrad import _numpy, _scipy  # noqa: F401, I100, I202

from pygrad._version import __version__  # noqa: F401, I202


_classes = [
    Config,
    Graph,
    Variable,
]


for _cls in _classes:
    _cls.__module__ = __name__


__all__ = ['config'] + [_cls.__name__ for _cls in _classes]


del _cls
del _classes
