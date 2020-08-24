import numpy as np
import pytest
from unittest.mock import patch

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


def normal(loc=None, scale=None, size=None):
    np.random.seed(0)
    sample = np.random.uniform(size=size)
    if scale is not None:
        sample *= np.asarray(scale)
    if loc is not None:
        loc += np.asarray(loc)
    return sample


@pytest.mark.parametrize('loc, scale, size, name', [
    (2, 1, None, 'normal'),
    ([1, -1, 5], [[2], [1]], None, 'normal'),
])
def test_forward(loc, scale, size, name):
    with patch('numpy.random.normal', side_effect=normal):
        actual = gd.random.normal(loc, scale, size, name=name)
    assert np.allclose(actual.data, normal(loc, scale, size))
    assert actual.name == name + '.out'


@pytest.mark.parametrize('loc, scale, size', [
    (
        gd.Array(np.random.rand(2, 3), is_variable=True),
        gd.Array(np.random.rand(2, 1), is_variable=True),
        None
    ),
    (
        gd.Array(np.random.rand(4, 2, 3), is_variable=True),
        gd.Array(np.random.rand(4, 1, 1), is_variable=True),
        None,
    ),
    (
        gd.Array(np.random.rand(2), is_variable=True),
        gd.Array(np.random.rand(2), is_variable=True),
        (3, 2)
    ),
])
def test_numerical_grad(loc, scale, size):
    with patch('numpy.random.normal', side_effect=normal):
        gd.random.normal(loc, scale, size).backward()
        dloc, dscale = _numerical_grad(
            lambda x, y: gd.random.normal(x, y, size), loc, scale)
    assert np.allclose(dloc, loc.grad, rtol=0, atol=1e-2)
    assert np.allclose(dscale, scale.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
