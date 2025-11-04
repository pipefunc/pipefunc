"""Nox configuration file."""

import os

import nox

nox.options.default_venv_backend = "uv"

python = ["3.11", "3.12", "3.13", "3.13t", "3.14", "3.14t"]
num_cpus = os.cpu_count() or 1
min_cpus = 2  # if ≤2 parallelization is not worth it
xdist = ("-n", "auto") if num_cpus > min_cpus else ()


@nox.session(python=python)
def pytest_min_deps(session: nox.Session) -> None:
    """Run pytest with no optional dependencies."""
    session.install(".[test]")
    session.run("pytest", *xdist)


@nox.session(python=python)
def pytest_all_deps(session: nox.Session) -> None:
    """Run pytest with "other" optional dependencies."""
    if session.python.endswith("t"):
        # Install all optional dependencies that work with 3.13t/3.14t
        extras = [
            "adaptive",
            "autodoc",
            "cli",
            "pandas",
            "plotting",
            "profiling",
            "pydantic",
            "rich",
            "widgets",
            "xarray",
            # Currently, all work except:
            # "polars",  # because polars-runtime-32 compiling takes long and fails
            # "mcp",  # because 'fastmcp' -> 'cryptography'
            # "zarr",  # because 'numcodecs' -> 'cryptography'
        ]
        session.install(f".[test,{','.join(extras)}]")
    else:
        session.install(".[all,test]")
    session.run("pytest", *xdist)
