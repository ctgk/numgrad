import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('parameters, learning_rate, error', [
    (
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([1., -1.], is_variable=True),
        ],
        1e-3, 'NoError',
    ),
    (
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([1., -1.], is_variable=True),
        ],
        -1e-3, ValueError,
    ),
])
def test_init_error(parameters, learning_rate, error):
    if error == 'NoError':
        gd.optimizers.Gradient(parameters, learning_rate)
    else:
        with pytest.raises(error):
            gd.optimizers.Gradient(parameters, learning_rate)


def test_minimize():
    theta = gd.Tensor([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Gradient([theta], learning_rate=0.1)
    for _ in range(100):
        theta.clear()
        d = theta - [-2, 4]
        loss = gd.square(d).sum()
        expected = theta.numpy() - 0.1 * 2 * d.numpy()
        optimizer.minimize(loss)
        assert np.allclose(theta.numpy(), expected)
    assert np.allclose(theta.numpy(), [-2, 4])


def test_maximize():
    theta = gd.Tensor([10, -20], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.Gradient([theta], learning_rate=0.1)
    for _ in range(100):
        theta.clear()
        d = theta - [-2, 4]
        score = -gd.square(d).sum()
        expected = theta.numpy() - 0.1 * 2 * d.numpy()
        optimizer.maximize(score)
        assert np.allclose(theta.numpy(), expected)
    assert np.allclose(theta.numpy(), [-2, 4])


if __name__ == "__main__":
    pytest.main([__file__])
