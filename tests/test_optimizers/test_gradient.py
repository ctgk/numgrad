import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('parameters, learning_rate, error', [
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 'NoError'
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        -1e-3, ValueError
    ),
])
def test_init_error(parameters, learning_rate, error):
    if error == 'NoError':
        gd.optimizers.Gradient(parameters, learning_rate)
    else:
        with pytest.raises(error):
            gd.optimizers.Gradient(parameters, learning_rate)


def test_minimize():
    theta = gd.Array([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Gradient([theta], learning_rate=0.1)
    for _ in range(100):
        with gd.Graph() as g:
            d = theta - [-2, 4]
            gd.square(d).sum()
        expected = theta.data - 0.1 * 2 * d.data
        optimizer.minimize(g)
        assert np.allclose(theta.data, expected)
    assert np.allclose(theta.data, [-2, 4])


def test_maximize():
    theta = gd.Array([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Gradient([theta], learning_rate=0.1)
    for _ in range(100):
        with gd.Graph() as g:
            d = theta - [-2, 4]
            -gd.square(d).sum()
        expected = theta.data - 0.1 * 2 * d.data
        g.backward()
        optimizer.maximize(clear_grad=False)
        theta.clear_grad()
        assert np.allclose(theta.data, expected)
    assert np.allclose(theta.data, [-2, 4])


if __name__ == "__main__":
    pytest.main([__file__])
