import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode
from skimage.transform import rotate
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pygrad as gd


class Encoder(gd.Module):

    def __init__(self, n_classes: int = 10, temperature: float = 0.1):
        super().__init__()
        self.layers = gd.nn.Sequential(
            gd.nn.Conv2D(1, 20, 5, strides=2),
            gd.nn.ReLU(),
            gd.nn.Conv2D(20, 20, 5, strides=2),
            gd.nn.ReLU(),
            gd.nn.Flatten(),
            gd.nn.Dense(4 * 4 * 20, 100),
            gd.nn.ReLU(),
        )
        self.c = gd.nn.Dense(100, n_classes, bias=False)
        self.m = gd.nn.Dense(100, 2, bias=False)
        self.s = gd.nn.Dense(100, 2)
        self.temperature = temperature

    def __call__(self, x):
        h = self.layers(x)
        y = gd.distributions.RelaxedCategorical(
            self.c(h), self.temperature, notation='q(y)')
        z = gd.distributions.Normal(
            self.m(h), gd.nn.softplus(self.s(h)), notation='q(z)')
        return y, z


class Decoder(gd.distributions.Distribution):

    def __init__(self, n_classes: int = 10):
        super().__init__('p(x|y,z)')
        self.dy = gd.nn.Dense(n_classes, 100, bias=False)
        self.dz = gd.nn.Dense(2, 100)
        self.layers = gd.nn.Sequential(
            gd.nn.ReLU(),
            gd.nn.Dense(100, 4 * 4 * 20),
            gd.nn.ReLU(),
            gd.nn.Reshape((-1, 4, 4, 20)),
            gd.nn.Conv2DTranspose(20, 20, 5, strides=2, shape=(12, 12)),
            gd.nn.ReLU(),
            gd.nn.Conv2DTranspose(20, 1, 5, strides=2, shape=(28, 28)),
        )

    def forward(self, y, z) -> gd.distributions.Bernoulli:
        h = self.dy(gd.log(y + 1e-5)) + self.dz(z)
        return gd.distributions.Bernoulli(self.layers(h))


class UVAE(gd.Module):

    def __init__(self, n_classes: int = 10, temperature: float = 1e-1):
        super().__init__()
        self.encoder = Encoder(n_classes, temperature)
        self.decoder = Decoder(n_classes)
        self.py = gd.distributions.Categorical(
            np.ones(n_classes) / n_classes, notation='p(y)')
        self.pz = gd.distributions.Normal(0., 1., notation='p(z)')

    def __call__(self, x):
        qy, qz = self.encoder(x)
        return qy.logits, qz.loc

    def elbo(self, x1, x2):
        qy1, qz1 = self.encoder(x1)
        qy2, qz2 = self.encoder(x2)
        y1s = qy1.sample()
        z1s = qz1.sample()
        y2s = qy2.sample()
        z2s = qz2.sample()
        px11 = self.decoder({**y1s, **z1s})
        px12 = self.decoder({**y1s, **z2s})
        px21 = self.decoder({**y2s, **z1s})
        px22 = self.decoder({**y2s, **z2s})
        elbo = 0.25 * (
            px11.logp(x1).sum() + px12.logp(x2).sum()
            + px21.logp(x1).sum() + px22.logp(x2).sum()) / x1.shape[0]
        elbo += 0.5 * (
            self.py.logp(y1s).sum() + self.pz.logp(z1s).sum()
            - qy1.logp(y1s).sum() - qz1.logp(z1s).sum()
            + self.py.logp(y2s).sum() + self.pz.logp(z2s).sum()
            - qy2.logp(y2s).sum() - qz2.logp(z2s).sum()) / x1.shape[0]
        return elbo


def visualize_encoding(encoder, x, indices, figname):
    y, z = encoder(x)
    y = y.logits.data
    z = z.loc.data
    k = y.shape[1]
    y = np.argmax(y, axis=-1)
    for i in range(k):
        plt.subplot(2, 5, i + 1)
        for j in range(k):
            plt.scatter(
                z[y == i, 0][indices[y == i] == j],
                z[y == i, 1][indices[y == i] == j],
                c=f'C{j}', s=1, label=str(j))
        plt.gca().spines['left'].set_position('center')
        plt.gca().spines['bottom'].set_position('center')
        plt.gca().spines['right'].set_color('none')
        plt.gca().spines['top'].set_color('none')
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('left')
        v = mode(indices[y == i])[0]
        v = str(v[0]) if v.size > 0 else 'NA'
        plt.title(v)
    plt.savefig(figname)
    plt.close()


def visualize_decoding(generator: gd.distributions.Distribution, figname):
    z = np.asarray(np.meshgrid(
        np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))).T.reshape(-1, 2)
    samples = []
    for i in range(10):
        y = np.array([[i == j for j in range(10)]] * 25)
        x_gen = gd.stats.sigmoid(generator.forward(y, z).logits).data
        samples.append(x_gen.reshape(25, 28, 28))
    samples = np.asarray(samples).reshape(2, 5, 5, 5, 28, 28)
    for i in range(10 * (5 * 5)):
        plt.subplot(10, 5 * 5, i + 1)
        plt.axis('off')
        plt.imshow(
            samples[i // 125][(i % 25) // 5][i % 5][(i % 125) // 25],
            cmap='gray', aspect='auto')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    plt.savefig(figname)
    plt.close()


def rotate_random(imgs):
    return np.asarray(
        [rotate(img, np.random.uniform(-40, 40)) for img in imgs],
        dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epoch', type=int, default=10, help='by default 10')
    parser.add_argument(
        '-b', '--batch', type=int, default=50, help='by default 50')
    parser.add_argument(
        '--random_seed', type=int, default=None, help='by default None')
    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
    gd.config.dtype = gd.Float32

    x, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
    x = (x > 127).reshape(-1, 28, 28, 1).astype(np.float32)
    y = y.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=10000, stratify=y)

    uvae = UVAE()
    optimizer = gd.optimizers.Adam(uvae)
    for e in range(1, args.epoch + 1):
        pbar = tqdm(range(0, len(x_train), args.batch))
        elbos = []
        for i in pbar:
            uvae.clear()
            x1 = gd.Tensor(rotate_random(x_train[i: i + args.batch]))
            x2 = gd.Tensor(rotate_random(x_train[i: i + args.batch]))
            elbo = uvae.elbo(x1, x2)
            optimizer.maximize(elbo)
            if optimizer.n_iter % 10 == 0:
                elbos.append(elbo.data)
                pbar.set_description(f'Epoch={e:2}, ELBO={np.mean(elbos): g}')
        indices = np.random.permutation(len(x_train))
        x_train, y_train = x_train[indices], y_train[indices]
        visualize_encoding(
            uvae.encoder, x_train[:1000], y_train[:1000],
            f'uvae_encode_epoch{e:02}.png')
        visualize_decoding(uvae.decoder, f'uvae_decode_epoch{e:02}.png')

    visualize_encoding(uvae.encoder, x_test, y_test, 'uvae_encoding.png')
    print(confusion_matrix(y_test, np.argmax(uvae(x_test)[0].data, -1)))
