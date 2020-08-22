from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._manipulation._reshape import reshape


class Flatten(Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x: Array, **kwargs) -> Array:
        return reshape(x, (x.shape[0], -1))
