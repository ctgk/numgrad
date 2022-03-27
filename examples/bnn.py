import matplotlib.pyplot as plt
import numpy as np

import pygrad as gd


class Qw(gd.distributions.Distribution):

    def __init__(self, size, notation: str) -> None:
        super().__init__(notation)
        self.loc = gd.Tensor(
            np.zeros(size), dtype=gd.config.dtype, is_variable=True)
        self.lns = gd.Tensor(
            np.zeros(size) - 5, dtype=gd.config.dtype, is_variable=True)

    def forward(self) -> gd.distributions.Normal:
        return gd.distributions.Normal(self.loc, gd.exp(self.lns))


class Py(gd.distributions.Distribution):

    def __init__(self, notation: str) -> None:
        super().__init__(notation)

    def forward(self, x, w1, b1, w2, b2) -> gd.distributions.Normal:
        x = gd.tanh(x @ w1 + b1)
        x = x @ w2 + b2
        return gd.distributions.Normal(x, 0.1)


if __name__ == "__main__":
    x = np.linspace(-3, 3, 20)[:, None]
    y = np.cos(x) + np.random.uniform(-0.1, 0.1, size=x.shape)
    x_test = np.linspace(-5, 5, 100)[:, None]

    gd.config.dtype = gd.Float32
    prior = (
        gd.distributions.Normal(np.zeros((1, 10)), 1, notation='p(w1)')
        * gd.distributions.Normal(np.zeros(10), 1, notation='p(b1)')
        * gd.distributions.Normal(np.zeros((10, 1)), 1, notation='p(w2)')
        * gd.distributions.Normal(np.zeros(1), 1, notation='p(b2)')
    )
    py = Py('p(y|x, w1, b1, w2, b2)')
    p_joint = py * prior
    plt.scatter(x.ravel(), y.ravel())
    for _ in range(10):
        plt.plot(
            x_test.ravel(),
            py({'x': x_test, **prior.sample()})._loc.data.ravel(),
            color='r',
        )
    plt.show()

    q = (
        Qw((1, 10), notation='q(w1)') * Qw(10, 'q(b1)')
        * Qw((10, 1), notation='q(w2)') * Qw(1, notation='q(b2)')
    )
    optimizer = gd.optimizers.Adam(q, 0.1)
    for _ in range(1000):
        q.clear()
        elbo = p_joint.logp({'y': y, **q.sample()}, {'x': x}) + q.entropy()
        optimizer.maximize(elbo)
        if optimizer.n_iter % 100 == 0:
            optimizer.learning_rate *= 0.8
    plt.scatter(x.ravel(), y.ravel())
    for _ in range(10):
        plt.plot(
            x_test.ravel(),
            py({'x': x_test, **q.sample()}).loc.data.ravel(),
            color='r')
    plt.show()
