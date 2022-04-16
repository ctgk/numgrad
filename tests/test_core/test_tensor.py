import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('args, kwargs, error', [
    (
        (0,),
        {},
        None,
    ),
    (
        ([0, 1],),
        {},
        None,
    ),
    (
        (0, gd.Float16),
        {},
        None,
    ),
    (
        (0,),
        {'dtype': gd.Float32},
        None,
    ),
    (
        ('a',),
        {},
        TypeError,
    ),
    (
        (['a', 'b'],),
        {},
        ValueError,
    ),
    (
        (0,),
        {'is_variable': True},
        None,
    ),
    (
        (0,),
        {'is_variable': 1},
        TypeError,
    ),
    (
        (0,),
        {'name': 'a'},
        None,
    ),
    (
        (0,),
        {'name': 1},
        TypeError,
    ),
])
def test_init(args, kwargs, error):
    if error is None:
        gd.Tensor(*args, **kwargs)
    else:
        with pytest.raises(error):
            gd.Tensor(*args, **kwargs)


@pytest.fixture(params=[
    {
        'tensor': gd.Tensor(0),
        'dtype_expected': gd.config.dtype,
        'is_variable_expected': False,
        'name_expected': None,
    },
    {
        'tensor': gd.Tensor(0, gd.Int64),
        'dtype_expected': gd.Int64,
        'is_variable_expected': False,
        'name_expected': None,
    },
    {
        'tensor': gd.Tensor([0, 1], is_variable=True),
        'dtype_expected': gd.config.dtype,
        'is_variable_expected': True,
        'name_expected': None,
    },
    {
        'tensor': gd.Tensor(np.zeros(0, dtype=int), name='a'),
        'dtype_expected': gd.config.dtype,
        'is_variable_expected': False,
        'name_expected': 'a',
    },
])
def parameter(request):
    return request.param


def test_dtype(parameter):
    tensor = parameter['tensor']
    expected = parameter['dtype_expected']
    actual = tensor.dtype
    assert actual == expected


def test_is_variable(parameter):
    tensor = parameter['tensor']
    expected = parameter['is_variable_expected']
    actual = tensor.is_variable
    assert actual == expected


def test_name(parameter):
    tensor = parameter['tensor']
    expected = parameter['name_expected']
    actual = tensor.name
    assert actual == expected


def test_grad_none():
    assert gd.Tensor([0, 1, 2]).grad is None


def test_grad_value_error_0():
    @gd.differentiable_operator
    def identity(x):
        def grad(dout):
            return dout
        return x, grad

    a = gd.Tensor([0, 1, 2], is_variable=True)

    identity(a)
    with pytest.raises(ValueError):
        a.grad


def test_grad_value_error_1():
    @gd.differentiable_operator
    def identity(x):
        def grad(dout):
            return dout
        return x, grad

    a = gd.Tensor([0, 1, 2], is_variable=True)

    identity(a)
    identity(a).backward()
    with pytest.raises(ValueError):
        a.grad


def test_asarray():
    actual = np.asarray(gd.Tensor([1, 2, 3]))
    assert np.allclose([1, 2, 3], actual)


if __name__ == '__main__':
    pytest.main([__file__])
