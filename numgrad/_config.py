import typing as tp

import numpy as np


class _Singleton:

    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(_Singleton, cls).__new__(cls)
        return cls._instance


class Config(_Singleton):
    """Configuration of numgrad module."""

    def __init__(self):
        """Initialize configuration."""
        self._dtype = np.float64
        self._graph = None
        self._verbosity: int = 0
        self._patched_function: tp.Dict[
            callable, tp.Tuple[str, str, callable]] = {}
        self._registered_gradient_function: tp.Dict[callable, callable] = {}

    @property
    def dtype(self) -> type:
        """Return default data type used in this library.

        Returns
        -------
        type
            One of these: `np.float32`, `np.float64`
        """
        return self._dtype

    @dtype.setter
    def dtype(self, value: type):
        """Set default data type to use.

        Parameters
        ----------
        value : type {float, np.float32, np.float64}
            New default data type.
        """
        if value not in (np.float32, np.float64):
            raise ValueError(
                'Default data type must be either '
                f'`np.float32` or `np.float64`, not {value}')
        self._dtype = value


config = Config()
