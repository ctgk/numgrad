import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('parameters, learning_rate, rho, error', [
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 0.9, 'NoError',
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
        ],
        1e-3, 1.1, ValueError,
    ),
    (
        [
            gd.Array(1., is_variable=True),
            gd.Array([1., -1.], is_variable=True),
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
    theta = gd.Array([0, 0], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.RMSProp([theta], learning_rate=0.1, rho=0.1)
    for _ in range(100):
        with gd.Graph() as g:
            d = theta - [-2, 4]
            gd.square(d).sum()
        optimizer.minimize(g)
    assert np.allclose(theta.data, [-2, 4], rtol=0, atol=0.1)


def test_maximize():
    theta = gd.Array([0, 0], dtype=gd.Float64, is_variable=True)
    optimizer = gd.optimizers.RMSProp([theta], learning_rate=0.1, rho=0.5)
    for _ in range(100):
        with gd.Graph() as g:
            d = theta - [-2, 4]
            -gd.square(d).sum()
        g.backward()
        optimizer.maximize(clear_grad=False)
        theta.clear_grad()
    assert np.allclose(theta.data, [-2, 4], rtol=0, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
