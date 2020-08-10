import numpy as np
import pytest
from unittest.mock import patch

import pygrad as pg
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
        actual = pg.random.uniform(low, high, size, name=name)
    assert np.allclose(actual.value, uniform(low, high, size))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('low, high, size, name', [
    (pg.Array(-1), 2, None, 'uniform'),
    ([1, -1, 5], pg.Array([[2], [1]]), (4, 2, 3), 'uniform'),
])
def test_forward_2(low, high, size, name):
    with patch('numpy.random.uniform', side_effect=uniform):
        actual = pg.random.uniform(low, high, size, name=name)
    low = low.value if isinstance(low, pg.Array) else low
    high = high.value if isinstance(high, pg.Array) else high
    assert np.allclose(actual.value, uniform(low, high, size))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('low, high, size', [
    (
        pg.Array(np.random.rand(2, 3), is_differentiable=True),
        pg.Array(np.random.rand(2, 1), is_differentiable=True),
        None
    ),
    (
        pg.Array(np.random.rand(4, 2, 3), is_differentiable=True),
        pg.Array(np.random.rand(4, 1, 1), is_differentiable=True),
        None,
    ),
    (
        pg.Array(np.random.rand(2), is_differentiable=True),
        pg.Array(np.random.rand(2), is_differentiable=True),
        (3, 2)
    ),
])
def test_numerical_grad(low, high, size):
    with patch('numpy.random.uniform', side_effect=uniform):
        pg.random.uniform(low, high, size).backward()
        dlow, dhigh = _numerical_grad(
            lambda x, y: pg.random.uniform(x, y, size), low, high)
    assert np.allclose(dlow, low.grad, rtol=0, atol=1e-2)
    assert np.allclose(dhigh, high.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
