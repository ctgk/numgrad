from contextlib import contextmanager
import typing as tp

from pygrad._array import Array
from pygrad._type_check import _typecheck_args


class Optimizer(object):
    """Base optimizer class.
    """

    @_typecheck_args
    def __init__(self, parameters: tp.Iterable[Array]):
        if not all(p.is_differentiable for p in parameters):
            raise ValueError('All \'parameters\' must be differentiable.')
        if any(len(p._parents) != 0 for p in parameters):
            raise ValueError('All \'parameters\' must not have parent nodes.')
        self._parameters = parameters
        self._n_iter: int = 0
        assert(callable(getattr(self, '_update')))

    @property
    def n_iter(self) -> int:
        return self._n_iter

    @staticmethod
    def _check_in_range(
            name: str,
            value: float,
            min_: float = None,
            max_: float = None):
        if ((min_ is not None and value < min_)
                or (max_ is not None and value >= max_)):
            raise ValueError(
                f'Value of arg "{name}" must be in range [{min_}, {max_}), '
                f'but was {value}')

    @contextmanager
    def _increment_count_calc_grad_clear_grad(
            self,
            value: Array = None,
            clear_grad: bool = True):
        self._n_iter += 1
        if value is not None:
            value.backward()
        try:
            yield
        finally:
            if clear_grad:
                for p in self._parameters:
                    p.clear_grad()

    @_typecheck_args
    def minimize(self, loss: Array):
        raise NotImplementedError

    @_typecheck_args
    def maximize(self, score: Array):
        raise NotImplementedError
