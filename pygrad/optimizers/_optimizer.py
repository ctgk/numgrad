import abc
import typing as tp

from pygrad._core._module import Module
from pygrad._core._tensor import Tensor
from pygrad._utils._typecheck import _typecheck


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
        if isinstance(parameters, Module):
            self._module = parameters
            parameters = tuple(self._module.variables.values())
        if not all(p.is_variable for p in parameters):
            raise ValueError('All \'parameters\' must be variable.')
        if any(p._parent is not None for p in parameters):
            raise ValueError('All \'parameters\' must not have parent nodes.')
        self._parameters = parameters
        self._n_iter: int = 0
        assert(callable(getattr(self, '_update')))

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
