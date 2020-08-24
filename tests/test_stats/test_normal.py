import numpy as np
import pytest

import pygrad as gd


class Normal(gd.stats.Normal):

    def __init__(self, rv='x', name='N'):
        super().__init__(rv, name=name)
        self.loc = gd.Array(0, dtype=gd.Float32, is_variable=True)
        self.lns = gd.Array(0, dtype=gd.Float32, is_variable=True)

    def forward(self):
        return {'loc': self.loc, 'scale': gd.exp(self.lns)}


def test_normal_logpdf():
    n = Normal()
    x = gd.Array([0, 2], dtype=gd.Float32)
    optimizer = gd.optimizers.Gradient(n, 0.1)
    for _ in range(100):
        optimizer.maximize(n.logpdf(x))
    assert np.isclose(n.loc.data, 1)
    assert np.isclose(n.lns.data, 0, rtol=0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])