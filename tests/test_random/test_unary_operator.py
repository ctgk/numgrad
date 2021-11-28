from unittest.mock import patch

import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


def fixed_gumbel(size):
    np.random.seed(0)
    return np.random.gumbel(size=size)


def fixed_exponential(size):
    np.random.seed(0)
    return np.random.standard_exponential(size=size)


@pytest.fixture(params=[
    dict(data=[1, -1, 2], size=(2, 3)),
    dict(data=np.random.uniform(-1, 1, (4, 3, 2))),
])
def parameter_data(request):
    return request.param


@pytest.fixture(params=[
    {
        'function': gd.random.exponential,
        'patch_target': 'numpy.random.standard_exponential',
        'patch_side_effect': fixed_exponential,
    },
    {
        'function': gd.random.gumbel_sigmoid,
        'patch_target': 'numpy.random.gumbel',
        'patch_side_effect': fixed_gumbel,
    },
    {
        'function': gd.random.gumbel_softmax,
        'patch_target': 'numpy.random.gumbel',
        'patch_side_effect': fixed_gumbel,
    },
])
def parameter_function(request):
    return request.param


def test_backward(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    f = parameter_function['function']
    s = parameter_function.get('size', x.shape)
    return_value = parameter_function['patch_side_effect'](size=s)
    with patch(
        parameter_function['patch_target'],
        return_value=return_value,
    ):
        out = f(x, size=s)
        out.backward()
        dx = _numerical_grad(lambda a: f(a, size=s), x, epsilon=1e-3)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
