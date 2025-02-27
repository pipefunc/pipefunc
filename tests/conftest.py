import importlib.util
import warnings

collect_ignore_glob = []


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
skip_if_missing("ipywidgets", match="*_widgets*")
skip_if_missing("pydantic", match="*_cli*")
skip_if_missing("pydantic", match="*_pydantic*")
