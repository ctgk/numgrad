from unittest.mock import patch

import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


def fixed_uniform(low, high, size=None):
    np.random.seed(0)
    low = np.asarray(low)
    high = np.asarray(high)
    if isinstance(size, int):
        size = (size,)
    elif size is None:
        size = (1,)
    return np.random.rand(*size) * (high - low) + low


def fixed_normal(loc=None, scale=None, size=None):
    np.random.seed(0)
    sample = np.random.uniform(size=size)
    if scale is not None:
        sample *= np.asarray(scale)
    if loc is not None:
        loc += np.asarray(loc)
    return sample


@pytest.fixture(params=[
    {'x': [1, -1, 2], 'y': 3, 'size': (2, 3)},
    {
        'x': np.random.uniform(-1, 1, (2, 3)),
        'y': np.random.uniform(3, 4, (4, 2, 3)),
    },
])
def parameter_data(request):
    return request.param


@pytest.fixture(params=[
    {
        'function': gd.random.uniform,
        'patch_target': 'numpy.random.uniform',
        'patch_side_effect': fixed_uniform,
    },
    {
        'function': gd.random.normal,
        'patch_target': 'numpy.random.normal',
        'patch_side_effect': fixed_normal,
    },
])
def parameter_function(request):
    return request.param


def test_backward(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['x'], is_variable=True)
    y = gd.Tensor(parameter_data['y'], is_variable=True)
    s = parameter_data.get('size', None)
    f = parameter_function['function']
    with patch(
        parameter_function['patch_target'],
        side_effect=parameter_function['patch_side_effect'],
    ):
        out = f(x, y, size=s)
        out.backward()
        dx, dy = _numerical_grad(
            lambda a, b: f(a, b, size=s), x, y, epsilon=1e-3)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
    assert np.allclose(dy, y.grad, rtol=0, atol=1e-2)
