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
