import argparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pygrad as gd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-b', '--batch', type=int, default=50)
    args = parser.parse_args()

    gd.config.dtype = gd.Float32
    x, y = fetch_openml('mnist_784', return_X_y=True)
    x = x.astype(np.float32).reshape(-1, 28, 28, 1)
    y = y.astype(np.int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=10000, stratify=y)
    cnn = gd.nn.Sequential(
        gd.nn.Conv2D(1, 20, 5),
        gd.nn.MaxPool2D(2),
        gd.nn.ReLU(),
        gd.nn.Conv2D(20, 20, 5),
        gd.nn.MaxPool2D(2),
        gd.nn.ReLU(),
        gd.nn.Flatten(),
        gd.nn.Dense(4 * 4 * 20, 100),
        gd.nn.Dropout(),
        gd.nn.Dense(100, 10),
    )
    optimizer = gd.optimizers.Adam(cnn)
    for e in range(1, args.epoch + 1):
        pbar = tqdm(range(0, len(x_train), args.batch))
        tp = 0
        total = 0
        for i in pbar:
            x_batch = x_train[i: i + args.batch]
            y_batch = y_train[i: i + args.batch]
            logits = cnn(x_batch)
            loss = gd.stats.sparse_softmax_cross_entropy(y_batch, logits).sum()
            optimizer.minimize(loss)
            if optimizer.n_iter % 10 == 0:
                tp += np.sum(y_batch == np.argmax(logits.data, -1))
                total += args.batch
                pbar.set_description(
                    f'Epoch={e}, Accuracy={int(100 * tp / total)}%')
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

    y_pred = np.argmax(cnn(x_test, droprate=None).data, -1)
    print(confusion_matrix(y_test, y_pred))
