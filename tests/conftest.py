import importlib.util
import warnings

has_adaptive = importlib.util.find_spec("adaptive") is not None
has_zarr = importlib.util.find_spec("zarr") is not None
has_xarray = importlib.util.find_spec("xarray") is not None
has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None

collect_ignore_glob = []

if not has_adaptive:
    warnings.warn("adaptive not installed, skipping adaptive tests", stacklevel=2)
    collect_ignore_glob.append("*_adaptive*")
if not has_zarr:
    warnings.warn("zarr not installed, skipping zarr tests", stacklevel=2)
    collect_ignore_glob.append("*_zarr*")
if not has_xarray:
    warnings.warn("xarray not installed, skipping xarray tests", stacklevel=2)
    collect_ignore_glob.append("*_xarray*")
if not has_ipywidgets:
    warnings.warn("ipywidgets not installed, skipping ipywidgets tests", stacklevel=2)
    collect_ignore_glob.append("*_widgets*")
