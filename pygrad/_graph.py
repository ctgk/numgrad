from collections import namedtuple
import typing as tp

from pygrad._config import config


Node = namedtuple(
    'Node', ('result', 'function', 'inputs', 'kwargs'))


class Graph(object):
    """Computational graph."""

    def __init__(self):
        """Construct computational graph."""
        super().__init__()
        self._node_list: tp.List[Node] = []

    def __enter__(self) -> 'Graph':
        """Return new computation graph to construct.

        Returns
        -------
        Graph
            New computational graph to construct.

        Raises
        ------
        ValueError
            Another graph is under construction.
        """
        if config._graph is not None:
            raise ValueError('There is already a graph under construction')
        config._graph = self
        return self

    def __exit__(self, *args, **kwargs):
        """Exit from the graph under construction."""
        config._graph = None

    def _add_node(self, result, function, *inputs, **kwargs):
        if any(result is node.result for node in self._node_list):
            raise ValueError('The result already exists in the graph')
        self._node_list.append(Node(result, function, inputs, kwargs))
