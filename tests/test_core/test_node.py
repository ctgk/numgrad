import pytest

from pygrad._core._node import _Node


@pytest.mark.parametrize('args, error', [
    (
        tuple(),
        None,
    ),
    (
        (tuple(), 1),
        TypeError,
    ),
    (
        (list(), 'a'),
        TypeError,
    ),
    (
        ((_Node(),), 'a'),
        None,
    ),
    (
        ((1,), 'a'),
        TypeError,
    ),
    (
        ((_Node(),), 'a,'),
        ValueError,
    ),
])
def test_init(args, error):
    if error is None:
        _Node(*args)
    else:
        with pytest.raises(error):
            _Node(*args)


if __name__ == '__main__':
    pytest.main([__file__])
