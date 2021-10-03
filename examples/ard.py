import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import pygrad as gd


class Pa(gd.distributions.Exponential):

    def __init__(self, scale: float, rv: str):
        super().__init__(rv=rv, name='p')
        self.scale = scale

    def forward(self) -> gd.stats.Exponential:
        return gd.stats.Exponential(self.scale)


class Pw(gd.distributions.Normal):

    def __init__(self, rv: str, condition: str):
        super().__init__(rv=rv, name='p', conditions=[condition])

    def forward(self, alpha) -> gd.stats.Normal:
        return gd.stats.Normal(0, 1 / alpha)


class Py(gd.distributions.Normal):

    def __init__(self):
        super().__init__(rv='y', name='p')

    def forward(self, x, w1, b1, w2, b2) -> gd.stats.Normal:
        h = gd.tanh(x @ w1 + b1)
        return gd.stats.Normal(h @ w2 + b2, 0.1)


class Qa(gd.distributions.Exponential):

    def __init__(self, size, rv: str):
        super().__init__(rv=rv, name='q')
        self.s = gd.Array(np.ones(size), gd.config.dtype, is_variable=True)

    def forward(self) -> gd.stats.Exponential:
        return gd.stats.Exponential(gd.nn.softplus(self.s))


class Qw(gd.distributions.Normal):

    def __init__(self, size, rv: str):
        super().__init__(rv=rv, name='q')
        self.loc = gd.Array(np.zeros(size), gd.config.dtype, is_variable=True)
        self.s = gd.Array(
            np.zeros(size) - 5, gd.config.dtype, is_variable=True)

    def forward(self) -> gd.stats.Normal:
        return gd.stats.Normal(self.loc, gd.nn.softplus(self.s))


class ARD(gd.Module):

    def __init__(self, scale):
        super().__init__()
        self.prior = (
            Pw(rv='w1', condition='a_w1')
            * Pa(scale, rv='a_w1')
            * Pw(rv='b1', condition='a_b1')
            * Pa(scale, 'a_b1')
            * Pw(rv='w2', condition='a_w2')
            * Pa(scale, 'a_w2')
            * Pw(rv='b2', condition='a_b2')
            * Pa(scale, 'a_b2')
        )
        self.py = Py()
        self.q = (
            Qa((1, 10), rv='a_w1') * Qw((1, 10), rv='w1')
            * Qa(10, rv='a_b1') * Qw(10, rv='b1')
            * Qa((10, 1), rv='a_w2') * Qw((10, 1), rv='w2')
            * Qa(1, rv='a_b2') * Qw(1, rv='b2')
        )

    def __call__(self, x):
        sample = self.q.sample()
        return self.py(x=x, **sample).loc

    def elbo(self, x, y):
        sample = self.q.sample()
        return (
            self.py.logpdf(y, {'x': x, **sample})
            + self.prior.logpdf(sample)
            - self.q.logpdf(sample, use_cache=True)
        )


if __name__ == "__main__":
    x = np.linspace(-3, 3, 20)[:, None].astype(np.float32)
    y = np.cos(x) + np.random.uniform(-0.1, 0.1, size=x.shape).astype(
        np.float32)
    x_test = np.linspace(-5, 5, 100)[:, None].astype(np.float32)

    gd.config.dtype = gd.Float32
    model = ARD(scale=10)
    optimizer = gd.optimizers.Adam(model, 0.1)
    with gd.Graph() as g:
        elbo = model.elbo(x, y)
    pbar = trange(10000)
    elbo_ma = elbo.data
    for _ in pbar:
        g.forward()
        optimizer.maximize(g)
        elbo_ma = 0.9 * elbo_ma + 0.1 * elbo.data
        pbar.set_description(f'{elbo_ma: g}')
        if optimizer.n_iter % 100 == 0:
            optimizer.learning_rate *= 0.95
    plt.subplot(1, 2, 1)
    plt.scatter(x.ravel(), y.ravel())
    for _ in range(10):
        plt.plot(x_test.ravel(), model(x_test).data.ravel(), color='C1')
    plt.subplot(1, 2, 2)
    plt.plot(
        (
            [f'w1[{i}]' for i in range(10)] + [f'b1[{i}]' for i in range(10)]
            + [f'w2[{i}]' for i in range(10)] + ['b2[0]']
        ),
        sum(list(list(v.data.ravel())
                 for k, v in model.trainables.items() if 'Qw.loc' in k),
            start=[]))
    plt.xticks(rotation=90)
    plt.grid(alpha=0.4)
    plt.show()
