from collections import OrderedDict

import numpy as np

from pygrad._core._array import Array
from pygrad._core._config import config


class Graph(object):
    """Computational graph class."""

    def __init__(self):
        """Initialize graph object."""
        super().__init__()
        self._parent = OrderedDict()
        self._children = {}
        self._backward_counts = {}
        self._terminal = None

    def __enter__(self):
        """Create graph context."""
        if config._graph is not None:
            raise ValueError('There is already a graph under construction')
        config._graph = self
        return self

    def __exit__(self, *args, **kwargs):
        """Exit current graph context."""
        config._graph = None
        for arg, children in self._children.items():
            if len(children) == 0:
                if self._terminal is None:
                    self._terminal = arg
                else:
                    raise ValueError('Multiple terminal node in graph')
        if self._terminal is None:
            raise ValueError('The graph does not have a terminal node')

    def _add_array_op(self, array, op):
        for arg in op._args:
            if arg in self._children:
                self._children[arg].append(op)
            else:
                self._children[arg] = [op]
            self._backward_counts[arg] = 0
        self._backward_counts[array] = 0
        self._children[array] = []
        self._parent[array] = op
        array._graph = self

    def forward(self):
        """Run forward propagation through the computational graph."""
        for k in self._backward_counts:
            self._backward_counts[k] = 0
        for arr, op in self._parent.items():
            for arg in op._args:
                arg._num_backwards = 0
                arg._grad = None
            arr._num_backwards = 0
            arr._grad = None
            arr._data = op._forward_numpy(*tuple(a._data for a in op._args))

    def _backward(self, a: Array):
        if a not in self._parent:
            return
        op = self._parent[a]
        dargs = op.backward(a._grad)
        for darg, arg in zip(dargs, op._args):
            if darg is not None:
                if arg._grad is None:
                    arg._grad = darg
                else:
                    arg._grad += darg
                self._backward_counts[arg] += 1
                if self._backward_counts[arg] == len(self._children[arg]):
                    self._backward(arg)
                elif self._backward_counts[arg] > len(self._children[arg]):
                    raise ValueError()

    def backward(self):
        """Run backpropagation through the computational graph."""
        self._terminal._grad = np.ones_like(self._terminal._data)
        self._backward(self._terminal)
