import pytest

import pygrad as gd


def test_enter_exit():
    assert gd.config.graph is None
    with gd.Graph() as g:
        assert gd.config.graph is g
    assert gd.config.graph is None


def test_enter_exit_error():
    with gd.Graph():
        with pytest.raises(ValueError):
            with gd.Graph():
                pass


def test_automatic_operation_storing():
    a = gd.Tensor(-1., is_variable=True)
    with gd.Graph() as g:
        b = gd.square(a)
    assert len(g._operations) == 1
    assert g._operations[0]._child is b
    assert g._operations[0]._args[0] is a


def test_no_automatic_operation_storing():
    a = gd.Tensor(-1)
    with gd.Graph() as g:
        gd.square(a)
    assert len(g._operations) == 0


if __name__ == '__main__':
    pytest.main([__file__])
