import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pygrad as gd


class Encoder(gd.distributions.Distribution):

    def __init__(self):
        super().__init__(notation='q(z|x)')
        self.d1 = gd.nn.Dense(784, 256)
        self.d2 = gd.nn.Dense(256, 128)
        self.dm = gd.nn.Dense(128, 2)
        self.ds = gd.nn.Dense(128, 2)

    def forward(self, x) -> gd.distributions.Normal:
        x = gd.tanh(self.d1(x))
        x = gd.tanh(self.d2(x))
        return gd.distributions.Normal(
            loc=self.dm(x), scale=gd.exp(self.ds(x)))


class Decoder(gd.distributions.Distribution):

    def __init__(self):
        super().__init__(notation='p(x|z)')
        self.d1 = gd.nn.Dense(2, 128)
        self.d2 = gd.nn.Dense(128, 256)
        self.d3 = gd.nn.Dense(256, 784)

    def forward(self, z) -> gd.distributions.Bernoulli:
        z = gd.tanh(self.d1(z))
        z = gd.tanh(self.d2(z))
        return gd.distributions.Bernoulli(logits=self.d3(z))


class VAE(gd.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.pz = gd.distributions.Normal(0, 1, notation='p(z)')

    def __call__(self, x):
        qz = self.encoder({'x': x})
        return gd.stats.sigmoid(self.decoder({'z': qz.loc}).logits)

    def elbo(self, x) -> gd.Tensor:
        qz = self.encoder({'x': x})
        p_joint = self.decoder * self.pz
        return p_joint.logp({'x': x, **qz.sample()}) + qz.entropy().sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batch', type=int, default=50)
    args = parser.parse_args()

    gd.config.dtype = gd.Float32
    x, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
    x = (x > 127).astype(np.float32)
    y = y.astype(int)
    x_train, x_test, _, _ = train_test_split(x, y, test_size=10000, stratify=y)

    vae = VAE()
    optimizer = gd.optimizers.Adam(vae)
    for e in range(1, args.epoch + 1):
        pbar = tqdm(range(0, len(x_train), args.batch))
        total_elbo = 0
        total_count = 0
        for i in pbar:
            vae.clear()
            x = gd.Tensor(x_train[i: i + args.batch])
            elbo = vae.elbo(x)
            optimizer.maximize(elbo)
            if optimizer.n_iter % 10 == 0:
                total_elbo = total_elbo + elbo.data
                total_count += 1
                pbar.set_description(
                    f'Epoch={e}, ELBO={total_elbo / total_count: g}')
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]

    x_reconst = vae(x_test[:50]).data
    for i in range(50):
        plt.subplot(10, 10, 2 * i + 1)
        plt.axis('off')
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.subplot(10, 10, 2 * i + 2)
        plt.axis('off')
        plt.imshow(x_reconst[i].reshape(28, 28), cmap='gray')
    plt.show()

    z = np.asarray(np.meshgrid(
        np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))).T.reshape(-1, 2)
    x_gen = gd.stats.sigmoid(vae.decoder({'z': z}).logits).data
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.axis('off')
        plt.imshow(x_gen[i].reshape(28, 28), cmap='gray')
    plt.show()
