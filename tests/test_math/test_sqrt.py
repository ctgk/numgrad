import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    ([1, 21, 5], 'sqrt', np.sqrt([1, 21, 5])),
])
def test_forward(x, name, expected):
    actual = gd.sqrt(x, name=name)
    assert np.allclose(actual.data, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x', [
    gd.Array(np.random.rand(2, 3) + 1, is_variable=True),
    gd.Array(np.random.rand(4, 2, 3) + 5, is_variable=True),
])
def test_numerical_grad(x):
    gd.sqrt(x).backward()
    dx = _numerical_grad(gd.sqrt, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
