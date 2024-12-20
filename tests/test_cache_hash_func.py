import importlib.util
import os

import numpy as np
import pytest

from pipefunc.cache import _get_dependency_source_code, _get_external_dependencies, hash_func

has_numpy = importlib.util.find_spec("numpy") is not None


def test_get_dependency_source_code():
    def f():
        return 1

    def g():
        return f()

    source_code = _get_dependency_source_code(g, set())
    assert "def f():" in source_code
    assert "def g():" in source_code


def test_get_external_dependencies():
    def f():
        return np.array([1, 2, 3])

    dependencies = _get_external_dependencies(f)
    assert "numpy" in dependencies
    assert dependencies["numpy"] == np.__version__


def test_hash_func_different_functions():
    def f():
        return 1

    def g():
        return 2

    assert hash_func(f) != hash_func(g)


def test_hash_func_same_function():
    def f():
        return 1

    assert hash_func(f) == hash_func(f)


def test_hash_func_bound_args():
    def f(a, b=1):
        return a + b

    assert hash_func(f, bound_args={"b": 1}) != hash_func(f, bound_args={"b": 2})


@pytest.mark.skipif(not has_numpy, reason="requires numpy")
def test_hash_func_external_dependencies():
    def f():
        return np.array([1, 2, 3])

    def g():
        return np.array([4, 5, 6])

    # Hashes should be different initially because of different function bodies
    assert hash_func(f) != hash_func(g)

    # Change an external dependency (e.g., by using a different numpy function)
    def h():
        return np.array([1, 2, 3, 4])

    # Hashes should be different due to the changed dependency
    assert hash_func(f) != hash_func(h)


def test_hash_func_pipefunc_version(monkeypatch):
    def f():
        return 1

    original_hash = hash_func(f)

    # Mock pipefunc.__version__ to be a different version
    monkeypatch.setattr("pipefunc.__version__", "0.0.0")

    new_hash = hash_func(f)
    assert original_hash != new_hash  # Hash should change with different version


def test_hash_func_no_external_dependencies():
    def f():
        return 1

    dependencies = _get_external_dependencies(f)
    assert dependencies == {}
    assert hash_func(f) == hash_func(f)


def test_hash_func_recursive_function():
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1)

    source_code = _get_dependency_source_code(factorial, set())
    assert "def factorial(n):" in source_code


def test_hash_func_ignore_internal_functions():
    def internal_function():
        return "internal"

    def f():
        return internal_function()

    dependencies = _get_external_dependencies(f)
    assert "internal_function" not in dependencies


def test_hash_func_lambda():
    def f(x):
        return x + 1

    assert hash_func(f) == hash_func(f)


def test_hash_func_builtin_module():
    def f():
        return os.path.join("a", "b")  # noqa: PTH118

    dependencies = _get_external_dependencies(f)
    assert "os" not in dependencies


def test_hash_func_class_methods():
    class MyClass:
        def method(self, x):
            return x + 1

        @classmethod
        def class_method(cls, x):
            return x + 2

        @staticmethod
        def static_method(x):
            return x + 3

    obj = MyClass()
    assert hash_func(obj.method) == hash_func(obj.method)
    assert hash_func(MyClass.class_method) == hash_func(MyClass.class_method)
    assert hash_func(MyClass.static_method) == hash_func(MyClass.static_method)
