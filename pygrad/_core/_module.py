import abc
from collections import OrderedDict
import inspect
import typing as tp

from pygrad._core._array import Array


class Module(abc.ABC):

    def __init__(self):
        super().__init__()
        object.__setattr__(self, '_module_name', self.__class__.__name__)
        self._trainables = OrderedDict()

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
        if isinstance(value, Array) and value.is_variable:
            self._trainables[self._module_name + '.' + name] = value
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                self._add_if_trainable(name + '_' + str(i), v)
        elif isinstance(value, dict):
            for k, v in value.items():
                self._add_if_trainable(name + '_' + k, v)
        elif isinstance(value, Module):
            for k, v in value._trainables.items():
                self._trainables['.'.join((self._module_name, name, k))] = v

    def clear_grad(self):
        self._assert_init()
        for param in self._trainables.values():
            param.clear_grad()

    def trainables(self) -> tp.List[Array]:
        self._assert_init()
        return list(self._trainables.values())
