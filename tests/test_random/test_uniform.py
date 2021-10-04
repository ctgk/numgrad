from unittest.mock import patch

import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


def uniform(low, high, size=None):
    np.random.seed(0)
    low = np.asarray(low)
    high = np.asarray(high)
    if isinstance(size, int):
        size = (size,)
    elif size is None:
        size = (1,)
    return np.random.rand(*size) * (high - low) + low


@pytest.mark.parametrize('low, high, size, name', [
    (-1, 2, None, 'uniform'),
    ([1, -1, 5], [[2], [1]], (4, 2, 3), 'uniform'),
])
def test_forward_1(low, high, size, name):
    with patch('numpy.random.uniform', side_effect=uniform):
        actual = gd.random.uniform(low, high, size, name=name)
    assert np.allclose(actual.data, uniform(low, high, size))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('low, high, size, name', [
    (gd.Array(-1), 2, None, 'uniform'),
    ([1, -1, 5], gd.Array([[2], [1]]), (4, 2, 3), 'uniform'),
])
def test_forward_2(low, high, size, name):
    with patch('numpy.random.uniform', side_effect=uniform):
        actual = gd.random.uniform(low, high, size, name=name)
    low = low.data if isinstance(low, gd.Array) else low
    high = high.data if isinstance(high, gd.Array) else high
    assert np.allclose(actual.data, uniform(low, high, size))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('low, high, size', [
    (
        gd.Array(np.random.rand(2, 3), is_variable=True),
        gd.Array(np.random.rand(2, 1), is_variable=True),
        None,
    ),
    (
        gd.Array(np.random.rand(4, 2, 3), is_variable=True),
        gd.Array(np.random.rand(4, 1, 1), is_variable=True),
        None,
    ),
    (
        gd.Array(np.random.rand(2), is_variable=True),
        gd.Array(np.random.rand(2), is_variable=True),
        (3, 2),
    ),
])
def test_numerical_grad(low, high, size):
    with patch('numpy.random.uniform', side_effect=uniform):
        with gd.Graph() as g:
            gd.random.uniform(low, high, size)
        g.backward()
        dlow, dhigh = _numerical_grad(
            lambda x, y: gd.random.uniform(x, y, size), low, high)
    assert np.allclose(dlow, low.grad, rtol=0, atol=1e-2)
    assert np.allclose(dhigh, high.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
