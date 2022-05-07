import numpy as np
import pytest

import numgrad as ng


def test_differentiable():

    def custom_gradient(do, _o, _x):
        return 3 * do

    @ng.differentiable(custom_gradient)
    def twice(x):
        return 2 * x

    a = ng.Variable([4, 2])
    with ng.Graph() as g:
        b = twice(a)
    assert np.allclose(g.gradient(b, [a])[0], np.array([3, 3]))


if __name__ == '__main__':
    pytest.main([__file__])
