from collections import namedtuple
import typing as tp

import numpy as np
import numpy
import scipy.special  # noqa: F401

from numgrad._config import config
from numgrad._utils._isscalar import _isscalar
from numgrad._variable import Variable


Node = namedtuple('Node', ('result', 'function', 'inputs', 'kwargs'))


class Graph(object):
    """Computational graph that stores forward path to backprop through.

    Examples
    --------
    >>> x = ng.Variable(1)
    >>> with ng.Graph() as g:
    ...     # Here, numpy and scipy functions are differentiable
    ...     y = np.tanh(x)
    ...
    >>> y
    0.7615941559557649
    >>> g.gradient(y, [x])
    (0.41997434161402614,)
    >>>
    >>> with ng.Graph() as g:
    ...     y = np.isnan(x)
    ...
    >>> y
    False
    >>> g.gradient(y, [x])  # fails to differentiate through `np.isnan`
    Traceback (most recent call last):
    ...
    TypeError: `target` of `numgrad.Graph.gradient()` must ...
    """

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
        self._check_type_of_target_and_sources(target, sources)
        id2grad = {id(target): np.ones_like(target._data)}
        for node in reversed(self._node_list):
            self._can_backprop_node(node, id2grad)
            for x, vjp in zip(node.inputs, config._func2vjps[node.function]):
                if not isinstance(x, Variable):
                    continue
                dx = self._get_grads(vjp, node, id2grad)
                self._accumulate_grad(id2grad, dx, x)
        return tuple(id2grad.get(id(s), None) for s in sources)

    @staticmethod
    def _check_type_of_target_and_sources(target, sources):
        if not isinstance(target, Variable):
            raise TypeError(
                '`target` of `numgrad.Graph.gradient()` must be an instance '
                f'of `ng.Variable`, not {type(target)}')
        if not isinstance(sources, (tuple, list)):
            raise TypeError(
                '`sources` of `numgrad.Graph.gradient()` must be list or '
                f'tuple of numgrad.Variable, not {type(sources)}')
        for s in sources:
            if not isinstance(s, Variable):
                raise TypeError(
                    '`sources` of `numgrad.Graph.gradient()` must be list or '
                    'tuple of numgrad.Variable, '
                    f'but contained an instance of {type(s)}')

    @staticmethod
    def _can_backprop_node(node: Node, id2grad: dict):
        if id(node.result) not in id2grad:
            return False
        if node.function not in config._func2vjps:
            raise NotImplementedError(
                f'Cannot backprop through {node.function}, '
                'VJP of the function is not registered yet.')
        return True

    @staticmethod
    def _get_grads(vjp: callable, node: Node, id2grad: dict):
        if isinstance(node.result, tuple):
            dy = tuple(id2grad.get(id(r), None) for r in node.result)
        else:
            dy = id2grad[id(node.result)]
        dx = vjp(dy, node.result, *node.inputs, **node.kwargs)
        return dx

    @staticmethod
    def _postprocess_nan_and_type(dx, x):
        if np.any(np.isnan(x)):
            dx = np.where(np.isnan(x), config.dtype(0), dx)
        if _isscalar(x):
            dx = np.take(dx, 0)
        return dx

    @classmethod
    def _accumulate_grad(cls, id2grad: dict, dx, x):
        dx = cls._postprocess_nan_and_type(dx, x)
        if id(x) in id2grad:
            id2grad[id(x)] = id2grad[id(x)] + dx
        else:
            id2grad[id(x)] = dx
