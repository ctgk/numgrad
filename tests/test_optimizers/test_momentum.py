import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('parameters, learning_rate, momentum, error', [
    (
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([1., -1.], is_variable=True),
        ],
        1e-3, 0.9, 'NoError',
    ),
    (
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([1., -1.], is_variable=True),
        ],
        1e-3, 1.1, ValueError,
    ),
    (
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([1., -1.], is_variable=True),
        ],
        1e-3, -0.1, ValueError,
    ),
])
def test_init_error(parameters, learning_rate, momentum, error):
    if error == 'NoError':
        gd.optimizers.Momentum(parameters, learning_rate, momentum)
    else:
        with pytest.raises(error):
            gd.optimizers.Momentum(parameters, learning_rate, momentum)


def test_minimize():
    theta = gd.Tensor([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Momentum(
        [theta], learning_rate=0.1, momentum=0.1)
    for _ in range(100):
        theta.clear()
        d = theta - [-2, 4]
        loss = gd.square(d).sum()
        optimizer.minimize(loss)
    assert np.allclose(theta.numpy(), [-2, 4])


def test_maximize():
    theta = gd.Tensor([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Momentum(
        [theta], learning_rate=0.1, momentum=0.5)
    for _ in range(100):
        theta.clear()
        d = theta - [-2, 4]
        score = -gd.square(d).sum()
        optimizer.maximize(score)
    assert np.allclose(theta.numpy(), [-2, 4])


if __name__ == "__main__":
    pytest.main([__file__])
