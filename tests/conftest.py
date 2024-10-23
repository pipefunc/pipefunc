import importlib.util

has_adaptive = importlib.util.find_spec("adaptive") is not None
has_zarr = importlib.util.find_spec("zarr") is not None
has_xarray = importlib.util.find_spec("xarray") is not None
has_ipywidgets = importlib.util.find_spec("ipywidgets") is not None

collect_ignore_glob = []

if not has_adaptive:
    collect_ignore_glob.append("*_adaptive*")
if not has_zarr:
    collect_ignore_glob.append("*_zarr*")
if not has_xarray:
    collect_ignore_glob.append("*_xarray*")
if not has_ipywidgets:
    collect_ignore_glob.append("*_widgets*")
