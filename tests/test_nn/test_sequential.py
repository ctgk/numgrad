import pytest

import pygrad as gd


def test_sequential():
    model = gd.nn.Sequential(
        gd.nn.Conv2D(1, 5, 3),
        gd.nn.MaxPool2D(2),
        gd.nn.Flatten(),
        gd.nn.Dense(20, 1),
    )
    x = gd.random.normal(0, 1, size=(100, 6, 6, 1))
    y = gd.random.uniform(0, 1, size=(100, 1))
    parameters = tuple(model.trainables.values())
    assert len(parameters) == 4
    optimizer = gd.optimizers.Adam(parameters)
    loss_prev = None
    for _ in range(100):
        with gd.Graph() as g:
            loss = gd.stats.sigmoid_cross_entropy(y, model(x)).sum()
        optimizer.minimize(g)
        if loss_prev is not None:
            assert loss_prev > loss.data
        loss_prev = loss.data


if __name__ == "__main__":
    pytest.main([__file__])
