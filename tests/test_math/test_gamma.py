import numpy as np
import pytest

import pygrad as pg
from pygrad._utils._numerical_grad import _numerical_grad


np.random.seed(0)


@pytest.mark.parametrize('x', [
    pg.Array(np.random.uniform(1, 9, (2, 3)), is_differentiable=True),
])
def test_numerical_grad(x):
    pg.gamma(x).backward()
    dx = _numerical_grad(lambda x: pg.gamma(x), x)[0]
    print(dx)
    print(x.grad)
    assert np.allclose(dx, x.grad, rtol=0, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
