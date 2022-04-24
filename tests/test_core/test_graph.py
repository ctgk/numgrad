import numpy as np
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
    a = np.array(-1)
    with gd.Graph() as g:
        b = gd.square(a)
    assert len(g._operations) == 1
    assert g._operations[0]._child is b
    assert g._operations[0]._args[0] is a


def test_gradient():
    a = np.array(-1)
    with gd.Graph() as g:
        b = gd.square(a)
    grads = g.gradient(b, [a])
    assert len(grads) == 1
    assert np.allclose(grads[0], -2)


def test_gradient_multiple_sources():
    a = np.array(-1)
    b = np.array(1)
    with gd.Graph() as g:
        c = gd.add(a, b)
    grads = g.gradient(c, [a, b])
    assert len(grads) == 2
    assert np.allclose(grads[0], 1)
    assert np.allclose(grads[1], 1)


def test_gradient_multiple_terminal_nodes():
    a = np.array(-1)
    with gd.Graph() as g:
        b = gd.square(a)
        c = gd.multiply(2, a)  # noqa: F841
    grads = g.gradient(b, [a])
    assert len(grads) == 1
    assert np.allclose(grads[0], -2)


if __name__ == '__main__':
    pytest.main([__file__])
