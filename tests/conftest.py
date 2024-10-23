import importlib.util

import pytest


def pytest_collect_file(parent, path):
    if path.basename == "test_adaptive.py":
        has_adaptive = importlib.util.find_spec("adaptive") is not None
        if not has_adaptive:
            pytest.skip("adaptive not installed", allow_module_level=True)
