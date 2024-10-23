import importlib.util
import warnings

has_adaptive = importlib.util.find_spec("adaptive") is not None
has_zarr = importlib.util.find_spec("zarr") is not None
has_xarray = importlib.util.find_spec("xarray") is not None
has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None

collect_ignore_glob = []


def skip_if_missing(name: str, has: bool, match: str | None = None) -> None:  # noqa: FBT001
    if not has:
        warnings.warn(
            f"{name} not installed, skipping {name} tests",
            stacklevel=3,
            category=ImportWarning,
        )
        collect_ignore_glob.append(f"*_{name}*" if match is None else match)


skip_if_missing("adaptive", has_adaptive)
skip_if_missing("zarr", has_zarr)
skip_if_missing("xarray", has_xarray)
skip_if_missing("ipywidgets", has_ipywidgets, match="*_widgets*")
