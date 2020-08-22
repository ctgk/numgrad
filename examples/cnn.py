import argparse

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pygrad as pg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=5)
    parser.add_argument('-b', '--batch', type=int, default=50)
    args = parser.parse_args()

    pg.config.dtype = pg.Float32
    x, y = fetch_openml('mnist_784', return_X_y=True)
    x = x.astype(np.float32).reshape(-1, 28, 28, 1)
    y = y.astype(np.int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=10000, stratify=y)
    cnn = pg.nn.Sequential(
        pg.nn.Conv2D(1, 20, 5),
        pg.nn.MaxPool2D(2),
        pg.nn.ReLU(),
        pg.nn.Conv2D(20, 20, 5),
        pg.nn.MaxPool2D(2),
        pg.nn.ReLU(),
        pg.nn.Flatten(),
        pg.nn.Dense(4 * 4 * 20, 100),
        pg.nn.Dropout(),
        pg.nn.Dense(100, 10),
    )
    optimizer = pg.optimizers.Adam(cnn.trainables())
    for e in range(1, args.epoch + 1):
        pbar = tqdm(range(0, len(x_train), args.batch))
        tp = 0
        total = 0
        for i in pbar:
            x_batch, y_batch = x[i: i + args.batch], y[i: i + args.batch]
            logits = cnn(x_batch)
            loss = pg.stats.sparse_softmax_cross_entropy(y_batch, logits).sum()
            optimizer.minimize(loss)
            if optimizer.n_iter % 10 == 0:
                tp += np.sum(y_batch == np.argmax(logits.value, -1))
                total += args.batch
                pbar.set_description(
                    f'Epoch={e}, Accuracy={int(100 * tp / total)}%')

    y_pred = np.argmax(cnn(x_test, droprate=None).value, -1)
    print(confusion_matrix(y_test, y_pred))
