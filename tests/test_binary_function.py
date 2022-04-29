import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('function, x, y', [
    (lambda a, b: a + b, gd.Tensor([[1, 2]]), gd.Tensor([[1], [2]])),
    (np.add, gd.Tensor([[1, 2]]), gd.Tensor([[1], [2]])),
])
def test_gradient(function, x, y):
    with gd.Graph() as g:
        output = function(x, y)
    dx_actual, dy_actual = g.gradient(output, [x, y])
    dx_expected, dy_expected = _numerical_grad(function, x, y)
    assert np.allclose(dx_expected, dx_actual)
    assert np.allclose(dy_expected, dy_actual)
