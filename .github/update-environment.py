"""Update environment.yml from pyproject.toml."""
from __future__ import annotations

from typing import Iterable

import tomllib


def clean_deps(deps: Iterable[str]) -> list[str]:
    """Remove version constraints from dependencies."""
    return [dep.split(";", 1)[0] for dep in deps]


def generate_environment_yml(
    data: dict,
    sections: tuple[str, ...] = ("all", "test", "docs", "plotting"),
    default_packages: tuple[str, ...] = ("python", "mpich"),
) -> str:
    """Generate environment.yml from pyproject.toml."""
    env_yaml = (
        "# This file is generated from pyproject.toml"
        " using .github/update-environment.py\n"
    )
    env_yaml += "name: pipefunc\n\n"
    env_yaml += "channels:\n- conda-forge\n\n"
    env_yaml += "dependencies:\n"

    # Default packages
    for dep in clean_deps(default_packages):
        env_yaml += f"  - {dep}\n"

    # Required deps from pyproject.toml
    env_yaml += "  # from pyproject.toml\n"
    for dep in clean_deps(data["project"]["dependencies"]):
        env_yaml += f"  - {dep}\n"

    # Optional dependencies
    for group in data["project"]["optional-dependencies"]:
        if group not in sections:
            continue
        env_yaml += f"  # optional-dependencies: {group}\n"
        for dep in clean_deps(data["project"]["optional-dependencies"][group]):
            env_yaml += f"  - {dep}\n"

    return env_yaml


if __name__ == "__main__":
    # Load pyproject.toml
    with open("pyproject.toml") as f:  # noqa: PTH123
        data = tomllib.loads(f.read())

    # Generate environment.yml
    environment_yml = generate_environment_yml(data, sections=("test", "plotting"))

    # Save environment.yml
    with open("environment.yml", "w") as f:  # noqa: PTH123
        f.write(environment_yml)

    # Generate environment.yml
    environment_yml = generate_environment_yml(
        data,
        sections=("test", "docs", "plotting"),
    )

    # Save environment.yml
    with open("docs/environment.yml", "w") as f:  # noqa: PTH123
        f.write(environment_yml)
