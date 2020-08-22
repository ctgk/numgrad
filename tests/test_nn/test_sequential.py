import pytest

import pygrad as pg


def test_sequential():
    model = pg.nn.Sequential(
        pg.nn.Conv2D(1, 5, 3),
        pg.nn.MaxPool2D(2),
        pg.nn.Flatten(),
        pg.nn.Dense(20, 1),
    )
    x = pg.random.normal(0, 1, size=(100, 6, 6, 1))
    y = pg.random.uniform(0, 1, size=(100, 1))
    parameters = model.trainables()
    assert len(parameters) == 4
    optimizer = pg.optimizers.Adam(parameters)
    loss_prev = None
    for _ in range(100):
        loss = pg.stats.sigmoid_cross_entropy(y, model(x)).sum()
        optimizer.minimize(loss)
        if loss_prev is not None:
            assert loss_prev > loss.data
        loss_prev = loss.data


if __name__ == "__main__":
    pytest.main([__file__])
