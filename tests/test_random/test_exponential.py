from unittest.mock import patch

import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


def standard_exponential(size=None):
    np.random.seed(0)
    return np.random.uniform(size=size)


@pytest.mark.parametrize('scale, size, name', [
    (gd.Array(2, dtype=gd.Float64), None, 'exponential'),
    (gd.Array([1, 0.4, 10], dtype=gd.Float32), (5, 3), 'exponential'),
])
def test_forward(scale, size, name):
    with patch(
            'numpy.random.standard_exponential',
            side_effect=standard_exponential):
        actual = gd.random.exponential(scale, size, name=name)
    assert np.allclose(actual.data, scale.data * standard_exponential(size))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('scale, size', [
    (gd.Array(np.random.rand(2, 3) * 10, is_variable=True), None),
    (gd.Array(np.random.rand(5, 1, 6) * 10, is_variable=True), (5, 3, 6)),
])
def test_numerical_grad(scale, size):
    with patch(
            'numpy.random.standard_exponential',
            side_effect=standard_exponential):
        with gd.Graph() as g:
            gd.random.exponential(scale, size)
        g.backward()
        dscale = _numerical_grad(
            lambda x: gd.random.exponential(x, size), scale)[0]
    print(dscale)
    print(scale.grad)
    assert np.allclose(dscale, scale.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
