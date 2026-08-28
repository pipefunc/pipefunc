"""Example pipeline for a physics simulation."""  # noqa: INP001

from dataclasses import dataclass

import numpy as np

from pipefunc import Pipeline, pipefunc


@dataclass(frozen=True)
class Geometry:  # noqa: D101
    x: float
    y: float


@dataclass(frozen=True)
class Mesh:  # noqa: D101
    geometry: Geometry
    mesh_size: float


@dataclass(frozen=True)
class Materials:  # noqa: D101
    geometry: Geometry
    materials: list[str]


@dataclass(frozen=True)
class Electrostatics:  # noqa: D101
    mesh: Mesh
    materials: Materials
    voltages: list[float]


@pipefunc(output_name="geo")
def make_geometry(x: float, y: float) -> Geometry:
    """Create a geometry object with given dimensions.

    Args:
        x: Width of the geometry
        y: Height of the geometry

    Returns:
        The created geometry object

    """
    return Geometry(x, y)


@pipefunc(output_name=("mesh", "coarse_mesh"))
def make_mesh(
    geo: Geometry,
    mesh_size: float,
    coarse_mesh_size: float = 0.1,
) -> tuple[Mesh, Mesh]:
    """Create fine and coarse meshes for the given geometry.

    Args:
        geo: Geometry to mesh
        mesh_size: Cell size for the fine mesh
        coarse_mesh_size: Cell size for the coarse mesh

    Returns:
        A tuple containing:
            - The fine mesh
            - The coarse mesh

    """
    return Mesh(geo, mesh_size), Mesh(geo, coarse_mesh_size)


@pipefunc(output_name="materials")
def make_materials(geo: Geometry) -> Materials:
    """Create materials for the given geometry.

    Args:
        geo: Geometry to create materials for

    Returns:
        The created materials object

    """
    return Materials(geo, ["i", "j", "c"])


@pipefunc(output_name="electrostatics", mapspec="V_left[i], V_right[j] -> electrostatics[i, j]")
def run_electrostatics(
    mesh: Mesh,
    materials: Materials,
    V_left: float,  # noqa: N803
    V_right: float,  # noqa: N803
) -> Electrostatics:
    """Run electrostatics simulation with given boundary conditions.

    Args:
        mesh: Mesh to run simulation on
        materials: Material properties
        V_left: Left boundary voltage
        V_right: Right boundary voltage

    Returns:
        Results of electrostatics simulation

    """
    return Electrostatics(mesh, materials, [V_left, V_right])


@pipefunc(output_name="charge", mapspec="electrostatics[i, j] -> charge[i, j]")
def get_charge(electrostatics: Electrostatics) -> float:
    """Calculate charge from electrostatics results.

    Parameters
    ----------
    electrostatics
        Results from electrostatics simulation

    Returns
    -------
    float
        Calculated charge (sum of voltages)

    """
    # obviously not actually the charge; but we should return _some_ number that
    # is "derived" from the electrostatics.
    return sum(electrostatics.voltages)


# No mapspec: function receives the full 2D array of charges!
@pipefunc(output_name="average_charge")
def average_charge(charge: np.ndarray) -> float:
    """Calculate average charge across all simulations.

    :param charge: 2D array of charges from all simulations
    :return: Mean charge value
    """
    return np.mean(charge)


pipeline_charge = Pipeline(
    [make_geometry, make_mesh, make_materials, run_electrostatics, get_charge, average_charge],
)
