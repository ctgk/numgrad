import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x', [
    pg.Array(np.random.uniform(-10, 10, (2, 3)), is_differentiable=True),
    pg.Array(np.random.uniform(-10, 10, (4, 2, 3)), is_differentiable=True),
])
def test_numerical_grad(x):
    pg.sinh(x).backward()
    dx = _numerical_grad(pg.sinh, x)[0]
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
