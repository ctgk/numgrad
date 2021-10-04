from pygrad._core._array import Array
from pygrad._core._module import Module
from pygrad._manipulation._reshape import reshape


class Flatten(Module):
    """Flatten module."""

    def __init__(self):
        """Initialize module."""
        super().__init__()

    def __call__(self, x: Array, **kwargs) -> Array:
        """Return flattened array.

        Parameters
        ----------
        x : Array
            Input.

        Returns
        -------
        Array
            Flattened array.
        """
        return reshape(x, (x.shape[0], -1))
