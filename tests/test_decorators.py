import numpy as np
import pytest

import pygrad as gd


def test_differentiable():

    def custom_gradient(do, o, x):
        return 3 * do

    @gd.differentiable(custom_gradient)
    def twice(x):
        return 2 * x

    a = gd.Variable([4, 2])
    with gd.Graph() as g:
        b = twice(a)
    assert np.allclose(g.gradient(b, [a])[0], np.array([3, 3]))


if __name__ == '__main__':
    pytest.main([__file__])
