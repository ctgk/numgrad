import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x', [
    pg.Array(np.random.uniform(-10, 10, (2, 3)), is_variable=True),
    pg.Array(np.random.uniform(-10, 10, (4, 2, 3)), is_variable=True),
])
def test_numerical_grad(x):
    pg.tanh(x).backward()
    dx = _numerical_grad(pg.tanh, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
