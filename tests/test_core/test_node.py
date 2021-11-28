import pytest

from pygrad._core._node import _Node


@pytest.mark.parametrize('name, error', [
    (
        None,
        None,
    ),
    (
        1,
        TypeError,
    ),
    (
        'a',
        None,
    ),
    (
        'a,',
        ValueError,
    ),
])
def test_init(name, error):
    if error is None:
        _Node(name)
    else:
        with pytest.raises(error):
            _Node(name)


if __name__ == '__main__':
    pytest.main([__file__])
