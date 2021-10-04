import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('parameters, learning_rate, beta1, beta2, error', [
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 0.9, 0.999, 'NoError',
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 1.1, 0.999, ValueError,
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, -0.01, 0.999, ValueError,
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 0.9, 1.001, ValueError,
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 0.9, -0.001, ValueError,
    ),
])
def test_init_error(parameters, learning_rate, beta1, beta2, error):
    if error == 'NoError':
        gd.optimizers.Adam(parameters, learning_rate, beta1, beta2)
    else:
        with pytest.raises(error):
            gd.optimizers.Adam(parameters, learning_rate, beta1, beta2)


def test_minimize():
    theta = gd.Array([-1, 5], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Adam([theta], learning_rate=1.)
    for _ in range(1000):
        with gd.Graph() as g:
            d = theta - [-2, 4]
            gd.square(d).sum()
        optimizer.minimize(g)
    assert np.allclose(theta.data, [-2, 4])


def test_maximize():
    theta = gd.Array([-1, 5], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Adam([theta], learning_rate=1.)
    for _ in range(1000):
        with gd.Graph() as g:
            d = theta - [-2, 4]
            -gd.square(d).sum()
        g.backward()
        optimizer.maximize(clear_grad=False)
        theta.clear_grad()
    assert np.allclose(theta.data, [-2, 4])


if __name__ == "__main__":
    pytest.main([__file__])
