import numpy as np
import pytest

import pygrad as pg


@pytest.mark.parametrize('parameters, learning_rate, beta1, beta2, error', [
    (
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([1., -1.], is_differentiable=True),
        ],
        1e-3, 0.9, 0.999, 'NoError'
    ),
    (
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([1., -1.], is_differentiable=True),
        ],
        1e-3, 1.1, 0.999, ValueError
    ),
    (
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([1., -1.], is_differentiable=True),
        ],
        1e-3, -0.01, 0.999, ValueError
    ),
    (
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([1., -1.], is_differentiable=True),
        ],
        1e-3, 0.9, 1.001, ValueError
    ),
    (
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([1., -1.], is_differentiable=True),
        ],
        1e-3, 0.9, -0.001, ValueError
    ),
])
def test_init_error(parameters, learning_rate, beta1, beta2, error):
    if error == 'NoError':
        pg.optimizers.Adam(parameters, learning_rate, beta1, beta2)
    else:
        with pytest.raises(error):
            pg.optimizers.Adam(parameters, learning_rate, beta1, beta2)


def test_minimize():
    theta = pg.Array([-1, 5], dtype=pg.Float64, is_differentiable=True)
    optimizer = pg.optimizers.Adam([theta], learning_rate=1.)
    for _ in range(1000):
        d = theta - [-2, 4]
        loss = pg.square(d).sum()
        optimizer.minimize(loss)
    assert np.allclose(theta.value, [-2, 4])


def test_maximize():
    theta = pg.Array([-1, 5], dtype=pg.Float64, is_differentiable=True)
    optimizer = pg.optimizers.Adam([theta], learning_rate=1.)
    for _ in range(1000):
        d = theta - [-2, 4]
        score = -pg.square(d).sum()
        score.backward()
        optimizer.maximize(clear_grad=False)
        theta.clear_grad()
    assert np.allclose(theta.value, [-2, 4])


if __name__ == "__main__":
    pytest.main([__file__])
