import importlib.util
import warnings

import _pytest

collect_ignore_glob = []


def pytest_addoption(parser: _pytest.config.argparsing.Parser) -> None:
    # NOTE: This is a workaround for the fact that the pytest-timeout plugin
    # doesn't seem to work with pytest-codspeed.
    parser.addoption(
        "--disable-timeout",
        action="store_true",
        default=False,
        help="Disable the pytest-timeout plugin",
    )


def pytest_configure(config: _pytest.config.Config) -> None:
    if config.getoption("--disable-timeout"):
        plugin = config.pluginmanager.getplugin("timeout")
        assert plugin is not None
        config.pluginmanager.unregister(plugin)


def skip_if_missing(name: str, match: str | None = None) -> None:
    has = importlib.util.find_spec(name) is not None
    if not has:
        warnings.warn(
            f"{name} not installed, skipping {name} tests",
            stacklevel=3,
            category=ImportWarning,
        )
        collect_ignore_glob.append(f"*_{name}*" if match is None else match)


skip_if_missing("adaptive")
skip_if_missing("zarr")
skip_if_missing("xarray")
skip_if_missing("ipywidgets", match="*_widgets*")
skip_if_missing("pydantic")
