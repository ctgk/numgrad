from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike
from pygrad._manipulation._reshape import reshape


class Flatten(Module):
    """Flatten module."""

    def __init__(self):
        """Initialize module."""
        super().__init__()

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:  # noqa: D102
        return reshape(x, (x.shape[0], -1))
