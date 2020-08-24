import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('parameters, learning_rate, momentum, error', [
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 0.9, 'NoError'
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 1.1, ValueError
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, -0.1, ValueError
    ),
])
def test_init_error(parameters, learning_rate, momentum, error):
    if error == 'NoError':
        gd.optimizers.Momentum(parameters, learning_rate, momentum)
    else:
        with pytest.raises(error):
            gd.optimizers.Momentum(parameters, learning_rate, momentum)


def test_minimize():
    theta = gd.Array([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Momentum(
        [theta], learning_rate=0.1, momentum=0.1)
    for _ in range(100):
        d = theta - [-2, 4]
        loss = gd.square(d).sum()
        optimizer.minimize(loss)
    assert np.allclose(theta.data, [-2, 4])


def test_maximize():
    theta = gd.Array([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Momentum(
        [theta], learning_rate=0.1, momentum=0.5)
    for _ in range(100):
        d = theta - [-2, 4]
        score = -gd.square(d).sum()
        score.backward()
        optimizer.maximize(clear_grad=False)
        theta.clear_grad()
    assert np.allclose(theta.data, [-2, 4])


if __name__ == "__main__":
    pytest.main([__file__])
