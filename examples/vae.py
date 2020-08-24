import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pygrad as gd


class Encoder(gd.stats.Normal):

    def __init__(self, rv='z', name='N'):
        super().__init__(rv=rv, name=name)
        self.d1 = gd.nn.Dense(784, 256)
        self.d2 = gd.nn.Dense(256, 128)
        self.dm = gd.nn.Dense(128, 2)
        self.ds = gd.nn.Dense(128, 2)

    def forward(self, x):
        x = gd.tanh(self.d1(x))
        x = gd.tanh(self.d2(x))
        return {'loc': self.dm(x), 'scale': gd.exp(self.ds(x))}


class Decoder(gd.stats.Bernoulli):

    def __init__(self, rv='x', name='Bern'):
        super().__init__(rv=rv, name=name)
        self.d1 = gd.nn.Dense(2, 128)
        self.d2 = gd.nn.Dense(128, 256)
        self.d3 = gd.nn.Dense(256, 784)

    def forward(self, z):
        z = gd.tanh(self.d1(z))
        z = gd.tanh(self.d2(z))
        return {'logits': self.d3(z)}


class VAE(gd.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.pz = gd.stats.Normal('z')

    def __call__(self, x):
        z = self.encoder.sample(conditions={'x': x})['z']
        return gd.stats.sigmoid(self.decoder(z=z)['logits'])

    def elbo(self, x) -> gd.Array:
        z = self.encoder.sample(conditions={'x': x})
        p = self.decoder * self.pz
        return p.logpdf(obs={'x': x, **z}) - self.encoder.logpdf(
            obs=z, conditions={'x': x}, use_cache=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-b', '--batch', type=int, default=50)
    args = parser.parse_args()

    gd.config.dtype = gd.Float32
    x, y = fetch_openml('mnist_784', return_X_y=True)
    x = (x > 127).astype(np.float32)
    y = y.astype(np.int)
    x_train, x_test, _, _ = train_test_split(x, y, test_size=10000, stratify=y)

    vae = VAE()
    optimizer = gd.optimizers.Adam(vae)
    for e in range(1, args.epoch + 1):
        pbar = tqdm(range(0, len(x_train), args.batch))
        total_elbo = 0
        total_count = 0
        for i in pbar:
            x_batch = x_train[i: i + args.batch]
            elbo = vae.elbo(x_batch)
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
    x_gen = gd.stats.sigmoid(vae.decoder(z=z)['logits']).data
    for i in range(100):
        plt.subplot(10, 10, i + 1)
        plt.axis('off')
        plt.imshow(x_gen[i].reshape(28, 28), cmap='gray')
    plt.show()
