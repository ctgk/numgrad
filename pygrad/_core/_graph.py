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
        target: np.ndarray,
        sources: tp.Union[tp.List[np.ndarray], tp.Tuple[np.ndarray, ...]],
    ) -> tp.Tuple[np.ndarray]:
        """Return gradient of target with respect to each source.

        Parameters
        ----------
        target : np.ndarray
            Target to be differentiated.
        sources : tp.Union[tp.List[np.ndarray], tp.Tuple[np.ndarray, ...]]
            Source tensors to differentiated against.

        Returns
        -------
        tp.Tuple[np.ndarray]
            Gradients of target with respect to each source.
        """
        def get_grad_and_clear(a: Tensor):
            out = a._grad
            a.clear()
            return out

        child_id_and_grad: tp.Dict[int, np.ndarray] = {}
        child_id_and_grad[id(target)] = np.ones_like(target)
        for op in reversed(self._operations):
            child_id = id(op._child)
            if child_id not in child_id_and_grad:
                continue
            dargs = op._grad_func(child_id_and_grad[child_id])
            if not isinstance(dargs, tuple):
                dargs = (dargs,)
            for arg, darg in zip(op._args, dargs):
                if darg is None:
                    continue
                if id(arg) in child_id_and_grad:
                    child_id_and_grad[id(arg)] += darg
                else:
                    child_id_and_grad[id(arg)] = np.ones_like(arg) * darg
        return tuple(child_id_and_grad.get(id(s), None) for s in sources)
