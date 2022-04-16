import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('parameters, learning_rate, rho, error', [
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
def test_init_error(parameters, learning_rate, rho, error):
    if error == 'NoError':
        gd.optimizers.RMSProp(parameters, learning_rate, rho)
    else:
        with pytest.raises(error):
            gd.optimizers.RMSProp(parameters, learning_rate, rho)


def test_minimize():
    theta = gd.Tensor([0, 0], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.RMSProp([theta], learning_rate=0.1, rho=0.1)
    for _ in range(100):
        theta.clear()
        d = theta - [-2, 4]
        loss = gd.square(d).sum()
        optimizer.minimize(loss)
    assert np.allclose(theta.numpy(), [-2, 4], rtol=0, atol=0.1)


def test_maximize():
    theta = gd.Tensor([0, 0], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.RMSProp([theta], learning_rate=0.1, rho=0.5)
    for _ in range(100):
        theta.clear()
        d = theta - [-2, 4]
        score = -gd.square(d).sum()
        optimizer.maximize(score)
    assert np.allclose(theta.numpy(), [-2, 4], rtol=0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
