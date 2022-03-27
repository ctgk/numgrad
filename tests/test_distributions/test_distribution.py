import numpy as np
import pytest

import pygrad as gd


class FailToInitialize1(gd.distributions.Distribution):

    def __init__(self, notation: str = 'p(x)') -> None:
        super().__init__(notation)

    def forward(self):
        pass


class FailToInitialize2(gd.distributions.Distribution):

    def __init__(self, notation: str = 'p(x)') -> None:
        super().__init__(notation)

    def forward(self) -> gd.distributions.BaseDistribution:
        pass


class Normal(gd.distributions.Distribution):

    def __init__(self, notation: str = 'p(x)') -> None:
        super().__init__(notation)
        self.loc = gd.Tensor(0, dtype=gd.Float32, is_variable=True)
        self.lns = gd.Tensor(0, dtype=gd.Float32, is_variable=True)

    def forward(self) -> gd.distributions.Normal:
        return gd.distributions.Normal(loc=self.loc, scale=gd.exp(self.lns))


class ConditionedNormal(gd.distributions.Distribution):

    def __init__(self, notation: str = 'p(x|mu)') -> None:
        super().__init__(notation)

    def forward(self, mu) -> gd.distributions.Normal:
        return gd.distributions.Normal(loc=mu, scale=1.)


@pytest.mark.parametrize('class_, expected', [
    (FailToInitialize1, ValueError),
    (FailToInitialize2, TypeError),
    (Normal, None),
])
def test_init(class_, expected):
    if expected is None:
        class_()
    else:
        with pytest.raises(expected):
            class_()


def test_init_error_empty_name():
    with pytest.raises(ValueError):
        Normal('p()')

    with pytest.raises(ValueError):
        Normal('(x)')

    with pytest.raises(ValueError):
        ConditionedNormal('p(x|)')


def test_entropy():
    p = Normal()
    assert np.isclose(p().entropy().data, 0.5 * (1 + np.log(2 * np.pi)))


def test_logp():
    p = Normal()
    x = gd.Tensor([0, 2], dtype=gd.Float32)
    optimizer = gd.optimizers.Gradient(p, 0.1)
    for _ in range(100):
        p.clear()
        score = p.logp(x)
        optimizer.maximize(score)
    assert np.isclose(p.loc.data, 1)
    assert np.isclose(p.lns.data, 0, rtol=0, atol=1e-5)


def test_sample():
    p = Normal()
    p.sample()


def test_joint():
    p_mu = gd.distributions.Normal(loc=0, scale=1, notation='p(mu)')
    p_x = ConditionedNormal(notation='p(x|mu)')
    p = p_x * p_mu
    assert repr(p) == 'p(x|mu)p(mu)'

    with pytest.raises(ValueError):
        p_x * p_x


if __name__ == "__main__":
    pytest.main([__file__])
