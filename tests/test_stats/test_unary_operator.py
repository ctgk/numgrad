import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.fixture(params=[
    {'data': np.random.uniform(-1, 1, (3, 2))},
    {'data': np.random.uniform(-1, 1, (5, 2, 4))},
])
def parameter_data(request):
    return request.param


@pytest.fixture(params=[
    {'function': gd.stats.log_softmax},
    {'function': gd.stats.sigmoid},
    {'function': gd.stats.softmax},
])
def parameter_function(request):
    return request.param


def test_single_operation(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    f = parameter_function['function']
    f(x).backward()
    dx = _numerical_grad(f, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


def test_double_operation(parameter_data, parameter_function):
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    f = parameter_function['function']
    f(f(x)).backward()
    dx = _numerical_grad(lambda a: f(f(a)), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)
