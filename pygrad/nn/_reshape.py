import typing as tp

from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._manipulation._reshape import reshape
from pygrad._utils._typecheck import _typecheck


class Reshape(Module):

    @_typecheck()
    def __init__(self, newshape: tp.Iterable[tp.Union[int]]):
        super().__init__()
        self._newshape = newshape

    def __call__(self, x: Array, **kwargs) -> Array:
        return reshape(x, self._newshape)
