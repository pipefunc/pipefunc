import importlib.util
import warnings
from contextlib import suppress

collect_ignore_glob = []


with suppress(ImportError):
    # To avoid https://github.com/holoviz/pyviz_comms/issues/137
    # Remove once https://github.com/holoviz/pyviz_comms/pull/138 is merged
    import builtins

    from IPython import get_ipython

    builtins.get_ipython = get_ipython  # type: ignore[attr-defined]


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
skip_if_missing("griffe", match="*autodoc*")
skip_if_missing("zarr")
skip_if_missing("xarray")
skip_if_missing("ipywidgets", match="*_widget*")
skip_if_missing("pydantic", match="*_cli*")
skip_if_missing("pydantic", match="*_pydantic*")
