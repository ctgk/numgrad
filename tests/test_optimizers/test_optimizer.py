import pytest

import pygrad as gd


class CallableUpdate(gd.optimizers.Optimizer):

    def _update(self):
        raise NotImplementedError


class NonCallableUpdate(gd.optimizers.Optimizer):

    def __init__(self, parameters):
        self._update = None
        super().__init__(parameters)


@pytest.mark.parametrize('cls_, parameters, error', [
    (
        CallableUpdate,
        [
            gd.Array(1., is_variable=True),
            gd.Array([0, -1.], is_variable=True),
        ],
        None,
    ),
    (
        gd.optimizers.Optimizer,
        [
            gd.Array(1., is_variable=True),
            gd.Array([0, -1.], is_variable=True),
        ],
        AttributeError,
    ),
    (
        NonCallableUpdate,
        [
            gd.Array(1., is_variable=True),
            gd.Array([0, -1.], is_variable=True),
        ],
        AssertionError,
    ),
    (
        CallableUpdate,
        [
            gd.Array(1.),
            gd.Array([0, -1.], is_variable=True),
        ],
        ValueError,
    ),
    (
        CallableUpdate,
        [
            gd.Array(1., is_variable=True),
            gd.sum(gd.Array([0, -1.], is_variable=True)),
        ],
        None,
    ),
])
def test_init_error(cls_, parameters, error):
    if error is None:
        cls_(parameters)
    else:
        with pytest.raises(error):
            cls_(parameters)


if __name__ == "__main__":
    pytest.main([__file__])
