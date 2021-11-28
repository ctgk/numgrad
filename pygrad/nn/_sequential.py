import typing as tp

from pygrad._core._module import Module
from pygrad._core._tensor import Tensor, TensorLike


class Sequential(Module):
    """Sequence of layers."""

    def __init__(self, *layers: tp.Iterable[Module]):
        """Construct sequence of layers."""
        super().__init__()
        if any(not isinstance(layer, Module) for layer in layers):
            raise TypeError(
                'All args of Sequential.__init__() must be '
                'instance of pygrad.Module')
        self.layers = layers

    def __call__(self, x: TensorLike, **kwargs) -> Tensor:
        """Process the input through the sequence of the layers.

        Parameters
        ----------
        x : TensorLike
            Input tensor-like object.

        Returns
        -------
        Tensor
            Processed tensor
        """
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x
