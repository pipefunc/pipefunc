import hashlib
import importlib.util
import inspect
import os
import sys
import types
from dataclasses import dataclass

import pytest

from pipefunc.cache import (
    _extract_source_with_dependency_info,
    _get_file_hash,
    extract_source_with_dependency_info,
    hash_func,
)

has_numpy = importlib.util.find_spec("numpy") is not None


has_numpy = importlib.util.find_spec("numpy") is not None


def f1():
    return 1


def g1():
    return f1()


def test_get_dependency_source_code():
    source_code = extract_source_with_dependency_info(g1)
    assert "def f1(" in source_code
    assert "def g1(" in source_code


def f2():
    if has_numpy:
        import numpy as np

        return np.array([1, 2, 3])
    return [1, 2, 3]


def test_get_external_dependencies():
    source = extract_source_with_dependency_info(f2)
    if has_numpy:
        assert "numpy" in source


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


def test_hash_func_recursive_function():
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1)

    source_code = extract_source_with_dependency_info(factorial)
    assert "def factorial(" in source_code


def test_hash_func_ignore_closure():
    """Closures are not easily accessible so they will not be included."""

    def internal_function():
        return "internal"

    def f():
        return internal_function()

    source = extract_source_with_dependency_info(f)
    assert "def internal_function" not in source


def test_hash_func_lambda():
    f = lambda x: x + 1  # noqa: E731
    # Lambdas may not have retrievable source code;
    # we expect consistent (if limited) behavior.
    assert hash_func(f) == hash_func(f)


def test_hash_func_builtin_module():
    def f():
        return os.path.join("a", "b")  # noqa: PTH118

    source = extract_source_with_dependency_info(f)
    # Standard library modules should be skipped.
    assert "os-" not in source


# --- Tests for classes and methods ---
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


def test_hash_func_class():
    class MyTestClass:
        def method1(self, x):
            return x + 10

        def method2(self, y):
            return self.method1(y) * 2

    hash1 = hash_func(MyTestClass)

    # Create a modified version and verify the hash is different.
    class MyTestClassModified:
        def method1(self, x):
            return x + 20

        def method2(self, y):
            return self.method1(y) * 2

    hash2 = hash_func(MyTestClassModified)
    assert hash1 != hash2


def test_hash_func_dataclass():
    @dataclass
    class Data:
        a: int
        b: int

    hash1 = hash_func(Data)

    @dataclass
    class DataModified:
        a: int
        b: int = 0

    hash2 = hash_func(DataModified)
    assert hash1 != hash2


def test_hash_func_nested_class():
    class Outer:
        class Inner:
            def inner_method(self):
                return "inner"

        def outer_method(self):
            return self.Inner().inner_method()

    _hash_outer = hash_func(Outer)
    src = extract_source_with_dependency_info(Outer)
    # The source for Outer should include the nested class Inner.
    assert "class Inner" in src


def test_hash_func_method_on_instance():
    class A:
        def method(self):
            return 42

    a = A()
    hash_method = hash_func(a.method)
    hash_method_again = hash_func(a.method)
    assert hash_method == hash_method_again


def test_hash_func_external_dependency_in_class():
    # This test uses numpy if available.
    if not has_numpy:
        pytest.skip("requires numpy")
    import numpy as np

    class WithNumpy:
        def use_numpy(self):
            return np.array([1, 2, 3])

    src = extract_source_with_dependency_info(WithNumpy)
    # External dependency info for numpy should be included.
    assert "numpy" in src


def dummy_function():
    return 1


def test_memo_branch():
    """Test that if the object is already in the memo, an empty string is returned."""
    memo = {dummy_function}
    result = _extract_source_with_dependency_info(dummy_function, memo)
    assert result == ""


def test_getsource_exception(monkeypatch, recwarn):
    """Test the branch when inspect.getsource raises an exception."""

    def fake_getsource(obj):
        msg = "fake error"
        raise OSError(msg)

    monkeypatch.setattr(inspect, "getsource", fake_getsource)

    def dummy():
        return 1

    memo = set()
    result = _extract_source_with_dependency_info(dummy, memo)
    assert result == ""
    warn = recwarn.pop(UserWarning)
    assert "Could not get source code for" in str(warn.message)


def test_no_module(monkeypatch):
    """Test that if inspect.getmodule returns None, the function returns the source code alone."""

    def dummy():
        return 1

    memo = set()
    # Monkey-patch inspect.getmodule to always return None.
    monkeypatch.setattr(inspect, "getmodule", lambda obj, default=None: None)  # noqa: ARG005
    source = inspect.getsource(dummy)
    result = _extract_source_with_dependency_info(dummy, memo)
    assert result == source


def test_external_module_dependency(monkeypatch):
    """Test the branch that processes an external module dependency."""
    # Create a fake module that is not part of the standard library.
    dummy_module = types.ModuleType("dummy_mod")
    dummy_module.__version__ = "1.0"
    dummy_module.__file__ = "/fake/path/dummy_mod.py"

    # Ensure dummy_mod is not considered part of the standard library.
    monkeypatch.setattr(sys, "stdlib_module_names", set(sys.stdlib_module_names) - {"dummy_mod"})

    # Create a dummy function that references "dummy_mod".
    def dummy_func():
        return dummy_mod.some_attr  # dummy_mod is referenced here  # noqa: F821

    # Inject our fake module into the globals of dummy_func so that it appears in the module dict.
    dummy_func.__globals__["dummy_mod"] = dummy_module

    memo = set()
    result = _extract_source_with_dependency_info(dummy_func, memo)
    # Since the file does not exist, _get_file_hash returns an empty string.
    expected_line = "# dummy_mod-1.0-/fake/path/dummy_mod.py-\n"
    assert expected_line in result


# -----------------------------------------------------------------------------
# Tests for _get_file_hash
# -----------------------------------------------------------------------------


def test_get_file_hash_no_file(tmp_path):
    """Test that _get_file_hash returns an empty string if the file does not exist."""
    non_existent = tmp_path / "nonexistent.txt"
    result = _get_file_hash(str(non_existent))
    assert result == ""


def test_get_file_hash_exception(monkeypatch, tmp_path, recwarn):
    """Test that _get_file_hash returns an empty string and warns when file reading fails."""
    # Create a temporary file.
    file_path = tmp_path / "test.txt"
    file_path.write_text("content")

    # Monkey-patch open to always raise an exception.
    def fake_open(*args, **kwargs):
        msg = "fake error"
        raise Exception(msg)  # noqa: TRY002

    monkeypatch.setattr("builtins.open", fake_open)

    result = _get_file_hash(str(file_path))
    assert result == ""
    warn = recwarn.pop(UserWarning)
    assert "Could not read file" in str(warn.message)


def test_get_file_hash_normal(tmp_path):
    """Test that _get_file_hash returns the correct hash for an existing file."""
    file_path = tmp_path / "test.txt"
    content = "hello world"
    file_path.write_text(content)
    expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    result = _get_file_hash(str(file_path))
    assert result == expected_hash
