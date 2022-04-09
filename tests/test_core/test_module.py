import pytest

import pygrad as gd


def test_init_no_error():
    class Derived(gd.Module):
        def __init__(self):
            super().__init__()
            self.a = 1

        def __call__(self):
            pass

    Derived()


def test_init_error_no_call_method():
    class Derived(gd.Module):
        def __init__(self):
            self.a = 1

    with pytest.raises(TypeError):
        Derived()


def test_init_error_no_super_init():
    class Derived(gd.Module):
        def __init__(self):
            self.a = 1

        def __call__(self):
            pass

    with pytest.raises(RuntimeError):
        Derived()


def test_skip_duplicate_variable():
    class Derived(gd.Module):
        def __init__(self, a: gd.Tensor):
            super().__init__()
            self.a = a
            self.b = a

        def __call__(self):
            pass

    a = gd.Tensor(0., is_variable=True)
    module = Derived(a)
    assert len(module.variables) == 1
    assert id(module.variables['Derived.a']) == id(a)


def test_skip_duplicate_module():
    class Derived(gd.Module):
        def __init__(self, n: int = 0):
            super().__init__()
            if n == 0:
                return
            d = Derived()
            self.a = [d for _ in range(n)]

        def __call__(self):
            pass

    module = Derived(n=2)
    assert len(module._modules) == 1
    assert 'Derived.a_0' in module._modules


def test_skip_duplicate_variable_in_module():

    class Derived(gd.Module):
        def __init__(self, a: gd.Tensor):
            super().__init__()
            self.a = a

        def __call__(self):
            pass

    class Derived1(gd.Module):
        def __init__(self, a: gd.Tensor):
            super().__init__()
            self.a = Derived(a)
            self.b = a

        def __call__(self):
            pass

    class Derived2(gd.Module):
        def __init__(self, a: gd.Tensor):
            super().__init__()
            self.a = a
            self.b = Derived(a)

        def __call__(self):
            pass

    a = gd.Tensor(1., is_variable=True)
    module1 = Derived1(a)
    module2 = Derived2(a)
    assert len(module1.variables) == 1
    assert id(module1.variables['Derived1.a.Derived.a']) == id(a)
    assert len(module2.variables) == 1
    assert id(module2.variables['Derived2.a']) == id(a)


if __name__ == "__main__":
    pytest.main([__file__])
