import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

import pygrad as gd


class Pw(gd.distributions.Distribution):

    def __init__(self, notation: str):
        super().__init__(notation)

    def forward(self, a) -> gd.distributions.Normal:
        return gd.distributions.Normal(0, 1 / a)


class Py(gd.distributions.Distribution):

    def __init__(self):
        super().__init__(notation='p(y|x,w1,b1,w2,b2)')

    def forward(self, x, w1, b1, w2, b2) -> gd.distributions.Normal:
        h = gd.tanh(x @ w1 + b1)
        return gd.distributions.Normal(h @ w2 + b2, 0.1)


class Qa(gd.distributions.Distribution):

    def __init__(self, size, notation: str):
        super().__init__(notation=notation)
        self.s = gd.Tensor(np.ones(size), gd.config.dtype, is_variable=True)

    def forward(self) -> gd.distributions.Exponential:
        return gd.distributions.Exponential(scale=gd.nn.softplus(self.s))


class Qw(gd.distributions.Distribution):

    def __init__(self, size, notation: str):
        super().__init__(notation=notation)
        self.loc = gd.Tensor(np.zeros(size), gd.config.dtype, is_variable=True)
        self.s = gd.Tensor(
            np.zeros(size) - 5, gd.config.dtype, is_variable=True)

    def forward(self) -> gd.distributions.Normal:
        return gd.distributions.Normal(self.loc, gd.nn.softplus(self.s))


class ARD(gd.Module):

    def __init__(self, scale):
        super().__init__()
        self.prior = (
            Pw(notation='p(w1|a_w1)')
            * gd.distributions.Exponential(scale=scale, notation='p(a_w1)')
            * Pw(notation='p(b1|a_b1)')
            * gd.distributions.Exponential(scale=scale, notation='p(a_b1)')
            * Pw(notation='p(w2|a_w2)')
            * gd.distributions.Exponential(scale=scale, notation='p(a_w2)')
            * Pw(notation='p(b2|a_b2)')
            * gd.distributions.Exponential(scale=scale, notation='p(a_b2)')
        )
        self.py = Py()
        self.p_joint = self.py * self.prior
        self.q = (
            Qa((1, 10), notation='q(a_w1)') * Qw((1, 10), notation='q(w1)')
            * Qa(10, notation='q(a_b1)') * Qw(10, notation='q(b1)')
            * Qa((10, 1), notation='q(a_w2)') * Qw((10, 1), notation='q(w2)')
            * Qa(1, notation='q(a_b2)') * Qw(1, notation='q(b2)')
        )

    def __call__(self, x):
        return self.py({
            'x': x,
            **{k: v for k, v in self.q.sample().items() if k[0] != 'a'},
        }).loc

    def elbo(self, x, y):
        return (
            self.p_joint.logp({'y': y, **self.q.sample()}, {'x': x})
            + self.q.entropy()
        )


if __name__ == "__main__":
    x = np.linspace(-3, 3, 20)[:, None].astype(np.float32)
    y = np.cos(x) + np.random.uniform(-0.1, 0.1, size=x.shape).astype(
        np.float32)
    x_test = np.linspace(-5, 5, 100)[:, None].astype(np.float32)

    gd.config.dtype = gd.Float32
    model = ARD(scale=10)
    optimizer = gd.optimizers.Adam(model, 0.1)
    pbar = trange(10000)
    elbo = model.elbo(x, y)
    elbo_ma = elbo.data
    for _ in pbar:
        model.clear()
        elbo = model.elbo(x, y)
        optimizer.maximize(elbo)
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
        sum(
            list(
                list(v.data.ravel()) for k, v in model.variables.items()
                if 'Qw.loc' in k
            ),
            [],
        ),
    )
    plt.xticks(rotation=90)
    plt.grid(alpha=0.4)
    plt.show()
