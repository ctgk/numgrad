import pytest


import pygrad as gd


def test_init_error_1():
    class Derived(gd.Module):
        def __init__(self):
            self.a = 1

    with pytest.raises(TypeError):
        Derived()


def test_init_error_2():
    class Derived(gd.Module):
        def __init__(self):
            self.a = 1

        def __call__(self):
            pass

    with pytest.raises(RuntimeError):
        Derived()


def test_init_error_3():
    class Derived(gd.Module):
        def __init__(self):
            super().__init__()
            self.a = 1

        def __call__(self):
            pass

    Derived()


def test_init_error_4():
    class Derived(gd.Module):
        def __init__(self):
            super().__init__()
            a = gd.Array(1., is_variable=True)
            self.a = a
            self.b = a

        def __call__(self):
            pass

    with pytest.raises(ValueError):
        Derived()


def test_init_error_5():
    class Derived(gd.Module):
        def __init__(self, n: int = 0):
            super().__init__()
            if n == 0:
                return
            d = Derived()
            self.a = [d for _ in range(n)]

        def __call__(self):
            pass

    with pytest.raises(ValueError):
        Derived(n=2)


if __name__ == "__main__":
    pytest.main([__file__])
