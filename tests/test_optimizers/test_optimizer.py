import pytest

import pygrad as pg


class CallableUpdate(pg.optimizers.Optimizer):

    def _update(self):
        raise NotImplementedError


class NonCallableUpdate(pg.optimizers.Optimizer):

    def __init__(self, parameters):
        self._update = None
        super().__init__(parameters)


@pytest.mark.parametrize('cls_, parameters, error', [
    (
        CallableUpdate,
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([0, -1.], is_differentiable=True),
        ],
        'NoError',
    ),
    (
        pg.optimizers.Optimizer,
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([0, -1.], is_differentiable=True),
        ],
        AttributeError,
    ),
    (
        NonCallableUpdate,
        [
            pg.Array(1., is_differentiable=True),
            pg.Array([0, -1.], is_differentiable=True),
        ],
        AssertionError,
    ),
    (
        CallableUpdate,
        [
            pg.Array(1.),
            pg.Array([0, -1.], is_differentiable=True),
        ],
        ValueError
    ),
    (
        CallableUpdate,
        [
            pg.Array(1., is_differentiable=True),
            pg.sum(pg.Array([0, -1.], is_differentiable=True)),
        ],
        ValueError
    ),
])
def test_init_error(cls_, parameters, error):
    if error == 'NoError':
        cls_(parameters)
    else:
        with pytest.raises(error):
            cls_(parameters)


if __name__ == "__main__":
    pytest.main([__file__])
