import numpy as np
import pytest

import pygrad
from pygrad._types import _to_pygrad_type


@pytest.mark.parametrize('value, dtype, is_differentiable, name, error', [
    (1, pygrad.Float64, True, 'x', 'NoError'),
    (int, pygrad.Float64, True, 'x', TypeError),
    (1, int, True, 'x', TypeError),
    (1, pygrad.Float64, 1, 'x', TypeError),
    (1, pygrad.Float64, True, 2, TypeError),
])
def test_array_init_error(value, dtype, is_differentiable, name, error):
    if error == 'NoError':
        pygrad.Array(value, dtype, is_differentiable, name=name)
    else:
        with pytest.raises(error):
            pygrad.Array(value, dtype, is_differentiable, name=name)


@pytest.mark.parametrize('value, dtype, name', [
    (1, None, None),
    (1, pygrad.Int8, None),
    (1, pygrad.Int16, 'x'),
    (1, pygrad.Float32, None),
    (1, pygrad.Float64, 'X.Y'),
])
def test_array_repr(value, dtype, name):
    kwargs = {
        k: v for k, v in zip(('dtype', 'name'), (dtype, name))
        if v is not None
    }
    actual = pygrad.Array(value, **kwargs)
    expected = repr(np.asarray(value, dtype=dtype))
    if name is not None:
        expected = expected[:-1] + f', name={name})'
    assert repr(actual) == expected


@pytest.mark.parametrize('value, dtype', [
    (1, None),
    ('1', pygrad.Int8),
    ([-1, 10], pygrad.Float32),
])
def test_dtype(value, dtype):
    actual = pygrad.Array(
        value, **{k: v for k, v in zip(['dtype'], [dtype]) if v is not None})
    expected = np.asarray(value, dtype=dtype)
    assert actual.dtype == _to_pygrad_type(expected.dtype)


@pytest.mark.parametrize('value', [
    np.random.rand(2, 3),
    np.random.rand(9, 1, 2, 5),
])
def test_ndim(value):
    actual = pygrad.Array(value)
    assert actual.ndim == np.asarray(value).ndim


@pytest.mark.parametrize('value', [
    np.random.rand(2, 3),
    np.random.rand(9, 1, 2, 5),
])
def test_size(value):
    actual = pygrad.Array(value)
    assert actual.size == np.asarray(value).size


@pytest.mark.parametrize('value', [
    np.random.rand(2, 3),
    np.random.rand(9, 1, 2, 5),
])
def test_shape(value):
    actual = pygrad.Array(value)
    assert actual.shape == np.asarray(value).shape


@pytest.mark.parametrize('value, is_differentiable', [
    (np.random.rand(2, 3), True),
    (1, False),
])
def test_is_differentiable(value, is_differentiable):
    actual = pygrad.Array(value, is_differentiable=is_differentiable)
    assert actual.is_differentiable is is_differentiable


if __name__ == "__main__":
    pytest.main([__file__])
