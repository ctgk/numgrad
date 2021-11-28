import abc
import inspect
import typing as tp

from pygrad._core._tensor import Tensor


class Module(abc.ABC):
    """Module class that contains variables and has call method."""

    def __init__(self):
        """Initialize module object."""
        super().__init__()
        object.__setattr__(self, '_module_name', self.__class__.__name__)
        self._variables = {}
        self._modules = {}

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Call method."""
        pass

    def _assert_init(self):
        if not hasattr(self, '_module_name'):
            raise RuntimeError('Should call super().__init__() in __init__()')

    def __setattr__(self, name, value):
        """Set an attribute."""
        self._assert_init()
        if inspect.stack()[1][3] == '__init__':
            self._add_if_variable(name, value)
        object.__setattr__(self, name, value)

    def _add_if_variable(self, name, value):
        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                self._add_if_variable(name + '_' + str(i), v)
        elif isinstance(value, dict):
            for k, v in value.items():
                self._add_if_variable(name + '_' + k, v)
        elif isinstance(value, Tensor) and value.is_variable:
            if value in self._variables.values():
                raise ValueError('Duplicate assignment of parameter')
            if value not in self._variables.values():
                self._variables[self._module_name + '.' + name] = value
        elif isinstance(value, Module):
            if value in self._modules.values():
                raise ValueError('Duplicate assignment of module')
            self._modules[self._module_name + '.' + name] = value
            for k, v in value._variables.items():
                self._variables['.'.join((self._module_name, name, k))] = v

    def clear(self):
        """Clear gradient of variables."""
        self._assert_init()
        for param in self._variables.values():
            param.clear()
        for module in self._modules.values():
            module.clear()

    @property
    def variables(self) -> tp.Dict[str, Tensor]:
        """Return dictionary of variables.

        Returns
        -------
        tp.Dict[str, Tensor]
            Dictionary of variables.
        """
        self._assert_init()
        return self._variables
