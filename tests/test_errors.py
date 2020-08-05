import pytest

import pygrad


@pytest.mark.parametrize('error', [
    pygrad.DifferentiationError,
])
def test_error(error):
    with pytest.raises(error):
        raise error


if __name__ == "__main__":
    pytest.main([__file__])
