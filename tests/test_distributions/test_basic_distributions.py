import numpy as np
import pytest

import pygrad as gd


@pytest.mark.parametrize('p, expected', [
    (gd.distributions.Bernoulli(logits=0), np.log(2)),
    (gd.distributions.Bernoulli(logits=99999999999), 0),
    (gd.distributions.Categorical(logits=[0] * 5), np.log(5)),
    (gd.distributions.Exponential(scale=10), 1 + np.log(10)),
    (gd.distributions.Exponential(rate=0.1), 1 + np.log(10)),
    (gd.distributions.Normal(0, 2), 0.5 * (np.log(4) + 1 + np.log(2 * np.pi))),
    (gd.distributions.Normal(1, 2), 0.5 * (np.log(4) + 1 + np.log(2 * np.pi))),
])
def test_entropy(p: gd.distributions.BaseDistribution, expected):
    actual = p.entropy().data
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('p, observed, expected', [
    (gd.distributions.Bernoulli(logits=0), 1, -np.log(2)),
    (gd.distributions.Bernoulli(logits=0), 0.5, -np.log(2)),
    (gd.distributions.Bernoulli(logits=999999999999), 1, 0),
    (gd.distributions.Categorical(logits=[0, 0, 0]), [1, 0, 0], -np.log(3)),
    (gd.distributions.Exponential(rate=0.1), 2, np.log(0.1) - 0.1 * 2),
    (gd.distributions.Exponential(scale=10), 2, np.log(0.1) - 0.1 * 2),
])
def test_logp(p: gd.distributions.BaseDistribution, observed, expected):
    actual = p.logp(observed).data
    assert np.allclose(actual, expected)


if __name__ == '__main__':
    pytest.main([__file__])
