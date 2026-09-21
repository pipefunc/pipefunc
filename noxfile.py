"""Nox configuration file."""

import os

import nox

nox.options.default_venv_backend = "uv"

python = ["3.11", "3.12", "3.13", "3.13t", "3.14", "3.14t"]
num_cpus = os.cpu_count() or 1
min_cpus = 2  # if ≤2 parallelization is not worth it
xdist = ("-n", "auto") if num_cpus > min_cpus else ()


def _install(session: nox.Session, *requirements: str) -> None:
    """Install dependencies without expensive source builds on Python 3.13t."""
    if session.python == "3.13t":
        # Many packages no longer publish wheels for this experimental ABI.
        # Keep small source builds for dependencies that have no 3.13t wheels.
        session.install(
            "--only-binary=:all:",
            "--no-binary=pipefunc,cffi,pyyaml,tornado",
            *requirements,
        )
    else:
        session.install(*requirements)


@nox.session(python=python)
def pytest_min_deps(session: nox.Session) -> None:
    """Run pytest with no optional dependencies."""
    _install(session, ".[test]")
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
        _install(session, f".[test,{','.join(extras)}]")
    else:
        _install(session, ".[all,test]")
    session.run("pytest", *xdist)
