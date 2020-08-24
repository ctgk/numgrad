import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('x, droprate, name', [
    (np.random.normal(size=(2, 5)), 0.5, 'dropout'),
    (np.random.normal(size=(4, 3)), 0.2, 'dropout'),
])
def test_forward(x, droprate, name):
    actual = gd.nn.dropout(x, droprate, name=name)
    print(x)
    print(actual)
    for i in range(x.size):
        assert (
            np.isclose(
                x.ravel()[i] / (1 - droprate), actual.data.ravel()[i])
            or np.isclose(actual.data.ravel()[i], 0)
        )
    assert actual.name == name + '.out'


@pytest.mark.parametrize('x', [
    np.random.normal(size=(10, 10)),
    np.random.normal(size=(5, 20)),
])
def test_forward_2(x):
    actual = gd.nn.dropout(x, droprate=None)
    assert actual is x


@pytest.mark.parametrize('x, droprate', [
    (gd.Array(np.random.rand(2, 3), is_variable=True), 0.1),
    (gd.Array(np.random.rand(4, 2, 3), is_variable=True), 0.5),
])
def test_backward(x, droprate):
    y = gd.nn.dropout(x, droprate)
    y.backward()
    dx = x.grad
    for i in range(x.size):
        if np.isclose(y.data.ravel()[i], 0):
            assert np.isclose(dx.ravel()[i], 0)
        else:
            assert np.isclose(dx.ravel()[i], 1 / (1 - droprate))


if __name__ == "__main__":
    pytest.main([__file__])
