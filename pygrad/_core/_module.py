import abc
import inspect
import typing as tp

from pygrad._core._array import Array


class Module(abc.ABC):

    def __init__(self):
        super().__init__()
        object.__setattr__(self, '_module_name', self.__class__.__name__)
        self._trainables = {}
        self._modules = {}

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def _assert_init(self):
        if not hasattr(self, '_module_name'):
            raise RuntimeError('Should call super().__init__() in __init__()')

    def __setattr__(self, name, value):
        self._assert_init()
        if inspect.stack()[1][3] == '__init__':
            self._add_if_trainable(name, value)
        object.__setattr__(self, name, value)

    def _add_if_trainable(self, name, value):
        if isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                self._add_if_trainable(name + '_' + str(i), v)
        elif isinstance(value, dict):
            for k, v in value.items():
                self._add_if_trainable(name + '_' + k, v)
        elif isinstance(value, Array) and value.is_variable:
            if value in self._trainables.values():
                raise ValueError('Duplicate assignment of parameter')
            if value not in self._trainables.values():
                self._trainables[self._module_name + '.' + name] = value
        elif isinstance(value, Module):
            if value in self._modules.values():
                raise ValueError('Duplicate assignment of module')
            self._modules[self._module_name + '.' + name] = value
            for k, v in value._trainables.items():
                self._trainables['.'.join((self._module_name, name, k))] = v

    def clear(self):
        self._assert_init()
        for param in self._trainables.values():
            param.clear_grad()
        for module in self._modules.values():
            module.clear()

    @property
    def trainables(self) -> tp.Dict[str, Array]:
        self._assert_init()
        return self._trainables
