import typing as tp

from pygrad._core._types import DataType, Float64, Int64, _is_int
from pygrad._utils._typecheck import _typecheck


class Config:

    def __init__(self):
        self._dtype = Float64
        self._int = Int64
        self._graph = None

    @property
    def dtype(self) -> DataType:
        """Default data type
        """
        return self._dtype

    @dtype.setter
    @_typecheck()
    def dtype(self, value: tp.Type[DataType]):
        self._dtype = value

    @property
    def int(self) -> DataType:
        """Default int type
        """
        return self._int

    @int.setter
    @_typecheck()
    def int(self, type_: tp.Type[DataType]):
        if not _is_int(type_):
            raise TypeError(
                f'Arg \'type_\' of Config.int() must be int type, not {type_}')
        self._int = type_


config = Config()
