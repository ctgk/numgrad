import typing as tp

from pygrad._core._types import DataType, Float64
from pygrad._utils._typecheck import _typecheck


class Config:

    def __init__(self):
        self._dtype = Float64

    @property
    def dtype(self) -> DataType:
        """Default data type
        """
        return self._dtype

    @dtype.setter
    @_typecheck()
    def dtype(self, value: tp.Type[DataType]):
        self._dtype = value


config = Config()
