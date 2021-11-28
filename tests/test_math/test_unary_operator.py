import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.fixture(params=[
    {'data': np.random.uniform(-1, 1, (2, 3))},
    {'data': np.random.uniform(-1, 1, (4, 2, 3))},
    {'data': np.random.uniform(-10, 10, (2, 3)), 'large_domain': True},
    {'data': np.random.uniform(-10, 10, (4, 2, 3)), 'large_domain': True},
    {
        'data': np.random.uniform(0.1, 1, (2, 4, 5)),
        'positive_domain': True,
    },
    {
        'data': np.random.uniform(1, 9, (2, 3)),
        'large_domain': True,
        'positive_domain': True,
    },
])
def parameter_data(request):
    return request.param


@pytest.fixture(params=[
    {'function': gd.cos},
    {'function': gd.cosh, 'remove': 'large_domain'},
    {'function': gd.exp, 'remove': 'large_domain'},
    {'function': gd.gamma, 'only': 'positive_domain', 'exclude_double': True},
    {'function': gd.log, 'only': 'positive_domain', 'exclude_double': True},
    {'function': gd.negate},
    {'function': gd.sin},
    {'function': gd.sinh, 'remove': 'large_domain'},
    {'function': gd.sqrt, 'only': 'positive_domain'},
    {'function': gd.square},
    {'function': gd.tan, 'remove': 'large_domain'},
    {'function': gd.tanh},
])
def parameter_function(request):
    return request.param


def test_single_operation(parameter_data, parameter_function):
    if 'remove' in parameter_function:
        if parameter_function['remove'] in parameter_data:
            return
    if 'only' in parameter_function:
        if parameter_function['only'] not in parameter_data:
            return
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    f = parameter_function['function']
    y = f(x)
    y.backward()
    dx = _numerical_grad(f, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


def test_double_operation(parameter_data, parameter_function):
    if 'remove' in parameter_function:
        if parameter_function['remove'] in parameter_data:
            return
    if 'only' in parameter_function:
        if parameter_function['only'] not in parameter_data:
            return
    if 'exclude_double' in parameter_function:
        return
    x = gd.Tensor(parameter_data['data'], is_variable=True)
    f = parameter_function['function']
    y = f(f(x))
    y.backward()
    dx = _numerical_grad(lambda a: f(f(a)), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
