from collections import OrderedDict
import typing as tp

import numpy as np

from pygrad._core._config import config
from pygrad._core._differentiable_operator import _DifferentiableOperator
from pygrad._core._tensor import Tensor
from pygrad._utils._typecheck import _typecheck


class Graph(object):
    """Computational graph."""

    def __init__(self):
        """Construct computational graph."""
        super().__init__()
        self._operations: tp.List[_DifferentiableOperator] = []
        self._parent = OrderedDict()
        self._children = {}
        self._backward_counts = {}
        self._terminal = None

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

    def _backward(self, a: Tensor):
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

    @_typecheck()
    def gradient(
        self,
        target: Tensor,
        sources: tp.Union[tp.List[Tensor], tp.Tuple[Tensor, ...]],
    ) -> tp.List[Tensor]:
        """Return gradient of target with respect to each source.

        Parameters
        ----------
        target : Tensor
            Target to be differentiated.
        sources : tp.Union[tp.List[Tensor], tp.Tuple[Tensor, ...]]
            Source tensors to differentiated against.

        Returns
        -------
        tp.List[Tensor]
            Gradients of target with respect to each source.
        """
        def get_grad_and_clear(a: Tensor):
            out = a._grad
            a.clear()
            return out

        assert all(s._grad is None for s in sources)
        assert target._grad is None
        target._grad = np.ones_like(target._data)
        for op in reversed(self._operations):
            child = op._child
            if child._grad is None:
                continue
            op.backward(child._grad)
        return [get_grad_and_clear(s) for s in sources]
