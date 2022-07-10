from collections import namedtuple
import typing as tp
from typing import List, Tuple, Union

import numpy as np
import numpy
import scipy.special  # noqa: F401

from numgrad._config import config
from numgrad._utils._isscalar import _isscalar
from numgrad._utils._to_array import _to_array_or_number
from numgrad._variable import Variable


Node = namedtuple(
    'Node',
    ('vjps', 'result', 'function', 'inputs', 'kwargs'),
)


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
    >>> g.backward(y, x)
    0.41997434161402614
    >>>
    >>> with ng.Graph() as g:
    ...     y = np.isnan(x)
    ...
    >>> y
    False
    >>> g.backward(y, x)  # fails to differentiate through `np.isnan`
    Traceback (most recent call last):
    ...
    TypeError: `target` of `numgrad.Graph.gradient()` must ...
    """

    def __init__(self, **kwargs):
        """Construct computational graph."""
        super().__init__()
        self._node_list: tp.List[Node] = []
        self._parent_graph: tp.Optional[Graph] = None
        self._allow_multiple_graphs: bool = kwargs.get(
            '_allow_multiple_graphs', False)

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
            if (
                not self._allow_multiple_graphs
                or not config._graph._allow_multiple_graphs
            ):
                raise ValueError('There is already a graph under construction')
            self._parent_graph = config._graph
            # should be no need to patch functions here.
        else:
            for module, func, patched in config._patched_function.values():
                setattr(eval(module), func, patched)
        config._graph = self
        return self

    def __exit__(self, *args, **kwargs):
        """Exit from the graph under construction."""
        config._graph = self._parent_graph
        if config._graph is None:
            for (
                original,
                (module, func, _),
            ) in config._patched_function.items():
                setattr(eval(module), func, original)

    def _add_node(self, result, function, *inputs, **kwargs):
        if function not in config._func2vjps:
            raise NotImplementedError(
                f'Cannot backprop through {function}, '
                'VJP of the function is not registered yet.')
        if any(result is node.result for node in self._node_list):
            raise ValueError('The result already exists in the graph')

        if isinstance(function, np.ufunc):
            vjps = config._func2vjps[function](*tuple(
                _to_array_or_number(a) if i < function.nin else a
                for i, a in enumerate(inputs)
            ), **kwargs)
        else:
            vjps = config._func2vjps[function](*inputs, **kwargs)
        if callable(vjps):
            vjps = (vjps,)
        node = Node(vjps, result, function, inputs, kwargs)
        if config._verbosity > 0:
            print('Graph:', self, ', Node:', node)
        self._node_list.append(node)
        self._add_node_to_parents(node)

    def _add_node_to_parents(self, node: Node):
        if self._parent_graph is None:
            return
        if config._verbosity > 0:
            print('Graph:', self._parent_graph, ', Node:', node)
        self._parent_graph._node_list.append(node)
        self._parent_graph._add_node_to_parents(node)

    def backward(
        self,
        target: Variable,
        sources: Union[Variable, List[Variable], Tuple[Variable, ...]],
        *,
        target_grad: tp.Optional[tp.Union[np.number, np.ndarray]] = None,
    ) -> tp.Union[np.ndarray, tp.Tuple[np.ndarray, ...]]:
        """Return gradients propagated backward from target to each source.

        Parameters
        ----------
        target : Variable
            Target to be differentiated.
        sources : Union[Variable, List[Variable], Tuple[Variable, ...]]
            Source variable(s) to differentiated against.
        target_grad : tp.Optional[tp.Union[np.number, np.ndarray]]
            Gradient to propagate backward from target, by default None.

        Returns
        -------
        tp.Union[np.ndarray, tp.Tuple[np.ndarray, ...]]
            Gradient(s) propagated backward from target to each source.
        """
        if return_single := isinstance(sources, Variable):
            sources = (sources,)
        self._check_type_of_target_and_sources(target, sources)
        target_grad = self._preprocess_target_grad(target_grad, target)
        id2grad = {id(target): target_grad}
        for node in reversed(self._node_list):
            if not self._can_backprop_node(node, id2grad):
                continue
            for x, vjp in zip(node.inputs, node.vjps):
                if not self._is_variable_or_tuple_of_variable(x):
                    continue
                dx = self._get_grads(vjp, node, id2grad)
                self._accumulate_grad(id2grad, dx, x)
        grads = tuple(id2grad.get(id(s), None) for s in sources)
        if return_single:
            return grads[0]
        return grads

    @staticmethod
    def _check_type_of_target_and_sources(target, sources):
        if not isinstance(target, Variable):
            raise TypeError(
                '`target` of `numgrad.Graph.gradient()` must be an instance '
                f'of `ng.Variable`, not {type(target)}')
        if isinstance(sources, Variable):
            return
        if not isinstance(sources, (tuple, list)):
            raise TypeError(
                '`sources` of `numgrad.Graph.gradient()` must be an instance '
                'of Variable, list of Variable, or tuple of Variable, '
                f'not {type(sources)}')
        for s in sources:
            if not isinstance(s, Variable):
                raise TypeError(
                    '`sources` of `numgrad.Graph.gradient()` must be list or '
                    'tuple of numgrad.Variable, '
                    f'but contained an instance of {type(s)}')

    @staticmethod
    def _preprocess_target_grad(target_grad, target):
        if target_grad is None:
            return np.ones_like(target)
        g = np.asarray(target_grad, dtype=config.dtype) * np.ones_like(target)
        if g.shape != target.shape:
            raise ValueError(
                f'Incompatible target_grad.shape {target_grad.shape} '
                f'with target.shape {target.shape}')
        return g

    @staticmethod
    def _can_backprop_node(node: Node, id2grad: dict):
        if node.function not in config._func2vjps:
            raise NotImplementedError(
                f'Cannot backprop through {node.function}, '
                'VJP of the function is not registered yet.')
        if isinstance(node.result, (list, tuple)):
            return all(
                id(r) in id2grad for r in node.result
                if isinstance(r, Variable))
        return id(node.result) in id2grad

    @staticmethod
    def _is_variable_or_tuple_of_variable(x):
        if isinstance(x, Variable):
            return True
        if isinstance(x, (tuple, list)) and any(
                isinstance(a, Variable) for a in x):
            return True
        return False

    @staticmethod
    def _get_grads(vjp: callable, node: Node, id2grad: dict):
        if isinstance(node.result, tuple):
            dy = tuple(id2grad.get(id(r), None) for r in node.result)
        else:
            dy = id2grad[id(node.result)]
        dx = vjp(dy, node.result)
        return dx

    @staticmethod
    def _postprocess_nan_and_type(dx, x):
        if np.any(np.isnan(x)):
            dx = np.where(np.isnan(x), config.dtype(0), dx)
        if _isscalar(x) and not _isscalar(dx):
            dx = np.take(dx, 0)
        return dx

    @classmethod
    def _accumulate_grad(cls, id2grad: dict, dx, x):
        if isinstance(dx, (tuple, list)) and isinstance(x, (tuple, list)):
            for x_, dx_ in zip(x, dx):
                if isinstance(x_, Variable):
                    cls._accumulate_grad(id2grad, dx_, x_)
        else:
            dx = cls._postprocess_nan_and_type(dx, x)
            if id(x) in id2grad:
                id2grad[id(x)] = id2grad[id(x)] + dx
            else:
                id2grad[id(x)] = dx
