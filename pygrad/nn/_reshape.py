import typing as tp

from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._manipulation._reshape import reshape
from pygrad._utils._typecheck import _typecheck


class Reshape(Module):
    """Reshape module."""

    @_typecheck()
    def __init__(self, newshape: tp.Union[tp.List[int], tp.Tuple[int, ...]]):
        """Initialize reshape module.

        Parameters
        ----------
        newshape : tp.Union[tp.List[int], tp.Tuple[int, ...]]
            New shape to reshape to.
        """
        super().__init__()
        self._newshape = newshape

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:  # noqa: D102
        return reshape(x, self._newshape)
