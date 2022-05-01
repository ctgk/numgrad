import argparse

import numpy as np
import scipy.special as sp
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numflow as nf


def mlp(x, w1, b1, w2, b2):
    return np.tanh(x @ w1 + b1) @ w2 + b2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch', type=int, default=50)
    args = parser.parse_args()

    x, y = fetch_openml('mnist_784', return_X_y=True, as_frame=False)
    x = x.astype(float).reshape(-1, 28 * 28 * 1)
    y = y.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=10000, stratify=y)

    w1 = nf.Variable(np.random.normal(scale=0.01, size=(28 * 28 * 1, 100)))
    b1 = nf.Variable(np.random.normal(scale=0.01, size=100))
    w2 = nf.Variable(np.random.normal(scale=0.1, size=(100, 10)))
    b2 = nf.Variable(np.random.normal(scale=0.1, size=10))

    for e in tqdm(range(1, 1 + args.epoch)):
        for i in range(0, len(x_train), args.batch):
            x = x_train[i: i + args.batch]
            y = y_train[i: i + args.batch]

            with nf.Graph() as g:
                logits = mlp(x, w1, b1, w2, b2)
                log_probas = sp.log_softmax(logits, axis=-1)
                nll = np.mean(-log_probas[range(len(log_probas)), y])
            grads = g.gradient(nll, (w1, b1, w2, b2))
            for p, g in zip((w1, b1, w2, b2), grads):
                p -= g * 0.01

        indices = np.random.permutation(len(x_train))
        x_train, y_train = x_train[indices], y_train[indices]

    predictions = np.argmax(mlp(x_test, w1, b1, w2, b2), axis=-1)
    print(f'Accuracy = {np.mean(y_test == predictions) * 100:g}%')
    print('Ground Truth (row) x Prediction (col)')
    print(confusion_matrix(y_test, predictions))
