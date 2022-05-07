from collections import namedtuple
import typing as tp

import numpy as np
import numpy
import scipy.special  # noqa: F401

from numgrad._config import config
from numgrad._variable import _ndarray_args, _ndarray_kwargs, Variable


Node = namedtuple('Node', ('result', 'function', 'inputs', 'kwargs'))


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
        for module, func, patched in config._patched_function.values():
            setattr(eval(module), func, patched)
        return self

    def __exit__(self, *args, **kwargs):
        """Exit from the graph under construction."""
        config._graph = None
        for original, (module, func, _) in config._patched_function.items():
            setattr(eval(module), func, original)

    def _add_node(self, result, function, *inputs, **kwargs):
        if any(result is node.result for node in self._node_list):
            raise ValueError('The result already exists in the graph')
        self._node_list.append(Node(result, function, inputs, kwargs))

    def gradient(
        self,
        target: Variable,
        sources: tp.Union[tp.List[Variable], tp.Tuple[Variable, ...]],
    ) -> tp.Tuple[np.ndarray]:
        """Return gradients of target with respect to each source.

        Parameters
        ----------
        target : Variable
            Target to be differentiated.
        sources : tp.Union[tp.List[Variable], tp.Tuple[Variable, ...]]
            Source tensors to differentiated against.
        Returns
        -------
        tp.Tuple[np.ndarray]
            Gradients of target with respect to each source.
        """
        if not isinstance(target, Variable):
            raise TypeError(
                '`target` must be an instance of `ng.Variable`, '
                f'not {type(target)}')
        tensor_id_to_grad: tp.Dict[int, np.ndarray] = {}
        tensor_id_to_grad[id(target)] = np.ones_like(target._data)
        for node in reversed(self._node_list):
            if id(node.result) not in tensor_id_to_grad:
                continue
            if node.function not in config._registered_gradient_function:
                raise NotImplementedError(
                    f'Gradient of {node.function} is not registered yet.')
            dargs = config._registered_gradient_function[node.function](
                tensor_id_to_grad[id(node.result)],
                node.result._data,
                *_ndarray_args(*node.inputs),
                **_ndarray_kwargs(**node.kwargs),
            )
            if not isinstance(dargs, tuple):
                dargs = (dargs,)
            for x, dx in zip(node.inputs, dargs):
                if dx is None:
                    continue
                if id(x) in tensor_id_to_grad:
                    tensor_id_to_grad[id(x)] += dx
                elif isinstance(x, Variable):
                    tensor_id_to_grad[id(x)] = np.ones_like(x._data) * dx
        return tuple(tensor_id_to_grad.get(id(s), None) for s in sources)
