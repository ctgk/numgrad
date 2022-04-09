import abc
import typing as tp

import numpy as np

from pygrad._core._module import Module
from pygrad._core._tensor import Tensor
from pygrad._utils._typecheck import _typecheck


def _return_tuple_of_named_parameters(parameters):
    if isinstance(parameters, Module):
        for k, p in parameters.variables.items():
            if p._name is None:
                p._name = k
        return tuple(parameters.variables.values())
    elif isinstance(parameters, (list, tuple)):
        for i, p in enumerate(parameters):
            if p._name is None:
                p._name = f'parameter_{i}'
        return tuple(parameters)
    elif isinstance(parameters, dict):
        for k, p in parameters.items():
            if p._name is None:
                p._name = k
        return tuple(parameters.values())
    raise TypeError(
        'The type of arg `parameters` must be either '
        '`pygrad.Module`, `list[pygrad.Tensor]`, '
        '`tuple[pygrad.Tensor, ...]`, or `tp.Dict[str, pygrad.Tensor]`, '
        f'not {type(parameters)}')


def _raise_error_if_not_variable(parameters):
    for p in parameters:
        if p.is_variable is False:
            raise ValueError(
                f'Parameter ({p.name}) is constant '
                'so that unable to optimize it.')


def _raise_error_if_has_parent_node(parameters):
    for p in parameters:
        if p._parent is not None:
            raise ValueError(f'Parameter ({p.name}) have a parent node')


def _get_one_duplicate(
    parameters: tp.Union[tp.List[Tensor], tp.Tuple[Tensor]],
) -> tp.Union[Tensor, None]:
    id_list = [id(p) for p in parameters]
    _, index_list, count_list = np.unique(
        id_list, return_index=True, return_counts=True)
    for index, count in zip(index_list, count_list):
        if count != 1:
            return parameters[index]
    return None


def _raise_error_if_has_duplicates(parameters):
    duplicate = _get_one_duplicate(parameters)
    if duplicate is not None:
        raise ValueError(
            f'Multiple passes of the same variable ({duplicate.name})')


class Optimizer(abc.ABC):
    """Base optimizer class."""

    def __init__(
        self,
        parameters: tp.Union[Module, tp.List[Tensor], tp.Tuple[Tensor]],
    ):
        """Initialize optimizer object.

        Parameters
        ----------
        parameters : tp.Union[Module, tp.List[Tensor], tp.Tuple[Tensor]]
            Parameters to optimize.
        """
        self._parameters = _return_tuple_of_named_parameters(parameters)
        _raise_error_if_not_variable(self._parameters)
        _raise_error_if_has_parent_node(self._parameters)
        _raise_error_if_has_duplicates(self._parameters)
        self._n_iter: int = 0

    @abc.abstractmethod
    def _update(self):
        pass

    @property
    def n_iter(self) -> int:
        """Return number of optimization iteration carried out.

        Returns
        -------
        int
            Number of optimization iteration carried out.
        """
        return self._n_iter

    @staticmethod
    def _check_in_range(
        name: str,
        value: float,
        min_: float = None,
        max_: float = None,
    ):
        if ((min_ is not None and value < min_)
                or (max_ is not None and value >= max_)):
            raise ValueError(
                f'Value of arg "{name}" must be in range [{min_}, {max_}), '
                f'but was {value}')

    def _increment_count_calc_grad(
        self,
        leaf_node: Tensor,
    ):
        self._n_iter += 1
        leaf_node.backward()

    @_typecheck()
    def maximize(self, score: Tensor):
        """Make small update of each parameter to maximize the given score.

        Parameters
        ----------
        score : Tensor
            Tensor to maximize.
        """
        self._maximize(score)

    @_typecheck()
    def minimize(self, loss: Tensor):
        """Make small update of each parameter to minimize the given loss.

        Parameters
        ----------
        loss : Tensor
            Tensor to minimize.
        """
        self._minimize(loss)

    @abc.abstractmethod
    def _maximize(self, score: Tensor):
        pass

    @abc.abstractmethod
    def _minimize(self, loss: Tensor):
        pass
