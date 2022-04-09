import pytest

import pygrad as gd


class CallableUpdate(gd.optimizers.Optimizer):

    def _update(self):
        raise NotImplementedError

    def _maximize(self, score):
        raise NotImplementedError

    def _minimize(self, loss):
        raise NotImplementedError


class NonCallableUpdate(gd.optimizers.Optimizer):

    def __init__(self, parameters):
        self._update = None
        super().__init__(parameters)

    def _maximize(self, score):
        raise NotImplementedError

    def _minimize(self, loss):
        raise NotImplementedError


@pytest.mark.parametrize('cls_, parameters, error', [
    (
        CallableUpdate,
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([0, -1.], is_variable=True),
        ],
        None,
    ),
    (
        gd.optimizers.Optimizer,
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([0, -1.], is_variable=True),
        ],
        TypeError,
    ),
    (
        NonCallableUpdate,
        [
            gd.Tensor(1., is_variable=True),
            gd.Tensor([0, -1.], is_variable=True),
        ],
        TypeError,
    ),
    (
        CallableUpdate,
        [
            gd.Tensor(1.),
            gd.Tensor([0, -1.], is_variable=True),
        ],
        ValueError,
    ),
    (
        CallableUpdate,
        [
            gd.Tensor(1., is_variable=True),
            gd.sum(gd.Tensor([0, -1.], is_variable=True)),
        ],
        ValueError,
    ),
])
def test_init_error(cls_, parameters, error):
    if error is None:
        cls_(parameters)
    else:
        with pytest.raises(error):
            cls_(parameters)


def test_init_error_pass_same_variable():
    a = gd.Tensor(1., is_variable=True)
    with pytest.raises(ValueError):
        CallableUpdate([a, a])


def test_named_module():

    class Derived(gd.Module):
        def __init__(self):
            super().__init__()
            self.a = gd.Tensor(0., is_variable=True)

        def __call__(self):
            pass

    module = Derived()
    CallableUpdate(module)
    assert module.a.name == 'Derived.a'


def test_named_list():
    a = gd.Tensor(0., is_variable=True)
    b = gd.Tensor(1., is_variable=True)
    CallableUpdate([a, b])
    assert a.name == 'parameter_0'
    assert b.name == 'parameter_1'


def test_named_dict():
    a = gd.Tensor(0., is_variable=True)
    b = gd.Tensor(1., is_variable=True)
    CallableUpdate({'a': a, 'b': b})
    assert a.name == 'a'
    assert b.name == 'b'


if __name__ == "__main__":
    pytest.main([__file__])
