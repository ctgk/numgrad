import numpy as np
import pytest

import pygrad as gd
from pygrad._utils._numerical_grad import _numerical_grad


@pytest.mark.parametrize('x, name, expected', [
    ([1, -1, 5], 'logsumexp', np.log(np.sum(np.exp([1, -1, 5])))),
])
def test_forward(x, name, expected):
    actual = gd.logsumexp(x, name=name)
    assert np.allclose(actual.data, expected)
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x, axis, keepdims', [
    (
        gd.Array(np.random.rand(2, 3), is_variable=True),
        0, False,
    ),
    (
        gd.Array(np.random.rand(4, 2, 3), is_variable=True),
        (0, 2), True,
    ),
    (
        gd.Array(np.random.rand(4, 2, 3), is_variable=True),
        None, False,
    ),
])
def test_numerical_grad(x, axis, keepdims):
    gd.logsumexp(x, axis, keepdims).backward()
    dx = _numerical_grad(lambda a: gd.logsumexp(a, axis, keepdims), x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
