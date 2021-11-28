import typing as tp

from pygrad._core._dtypes import _is_int, DataType, Float64, Int64
from pygrad._utils._typecheck import _typecheck


class _Singleton:

    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(_Singleton, cls).__new__(cls)
        return cls._instance


class Config(_Singleton):
    """Configuration of pygrad module."""

    def __init__(self):
        """Initialize configuration."""
        self._dtype = Float64
        self._int = Int64
        self._graph = None

    @property
    def dtype(self) -> DataType:
        """Return default data type."""
        return self._dtype

    @dtype.setter
    @_typecheck()
    def dtype(self, value: tp.Type[DataType]):
        """Set default data type to use.

        Parameters
        ----------
        value : tp.Type[DataType]
            New default data type.
        """
        self._dtype = value

    @property
    def int(self) -> DataType:
        """Return default int type."""
        return self._int

    @int.setter
    @_typecheck()
    def int(self, type_: tp.Type[DataType]):
        """Set default integer data type."""
        if not _is_int(type_):
            raise TypeError(
                f'Arg \'type_\' of Config.int() must be int type, not {type_}')
        self._int = type_


config = Config()
