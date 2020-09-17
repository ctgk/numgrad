import numpy as np
import pytest

import pygrad as gd


class Normal(gd.distributions.Normal):

    def __init__(self, rv='x', name='N'):
        super().__init__(rv, name=name)
        self.loc = gd.Array(0, dtype=gd.Float32, is_variable=True)
        self.lns = gd.Array(0, dtype=gd.Float32, is_variable=True)

    def forward(self) -> gd.stats.Normal:
        return gd.stats.Normal(self.loc, gd.exp(self.lns))


def test_normal_logpdf():
    n = Normal()
    x = gd.Array([0, 2], dtype=gd.Float32)
    optimizer = gd.optimizers.Gradient(n, 0.1)
    for _ in range(100):
        with gd.Graph() as g:
            n.logpdf(x)
        optimizer.maximize(g)
    assert np.isclose(n.loc.data, 1)
    assert np.isclose(n.lns.data, 0, rtol=0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
