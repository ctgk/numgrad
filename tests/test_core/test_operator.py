import numpy as np
import pytest

import pygrad as gd


def test_decoration():

    @gd.differentiable_operator
    def identity(x):
        def grad(dout):
            return dout
        return x, grad
    a = gd.Tensor([-1, 0, 1], is_variable=True)
    b = identity(a)
    b.backward()
    assert np.allclose(a.grad, 1)


if __name__ == "__main__":
    pytest.main([__file__])
