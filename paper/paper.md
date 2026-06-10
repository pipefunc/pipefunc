---
title: 'pipefunc: Python pipelines as function compositions with declarative N-dimensional sweeps'
tags:
  - Python
  - workflows
  - pipelines
  - directed acyclic graphs
  - high-performance computing
  - parallel computing
  - parameter sweeps
authors:
  - name: Bas Nijholt
    orcid: 0000-0003-0383-4986
    affiliation: "1, 2"
affiliations:
  - name: Delft University of Technology, Delft, The Netherlands
    index: 1
  - name: Current address — IonQ Inc., College Park, Maryland, USA
    index: 2
date: 10 June 2026
bibliography: paper.bib
---

# Summary

Scientific computing frequently involves chaining together functions—simulate, post-process, aggregate, analyze—and running that chain over large parameter sweeps on anything from a laptop to a supercomputer.
`pipefunc` is a lightweight Python library that turns plain Python functions into such computational workflows.
By annotating functions with a decorator that names their outputs, `pipefunc` assembles them into a directed acyclic graph (DAG), automatically resolves execution order, validates type annotations across function boundaries, and executes the graph—either serially or in parallel—without requiring the user to write any orchestration code.

The central abstraction is the *mapspec*, a declarative, einsum-like index notation that specifies how functions map over N-dimensional input spaces.
For example, `"a[i], b[j] -> c[i, j]"` declares that a function is evaluated over the outer product of `a` and `b`, producing a two-dimensional result array.
From the mapspecs of all functions in a pipeline, `pipefunc` infers array shapes, derives which elements can be computed independently, schedules them in parallel, and—optionally, through the Xarray integration—assembles outputs into labeled multi-dimensional arrays [@hoyer2017xarray].
Map, zip, outer-product, reduction, and dynamic-shape patterns are all expressed in this single notation, eliminating the bookkeeping loops and index arithmetic that otherwise dominate sweep code.

# Statement of need

In practice, many computational studies (e.g., simulating a physical device over a grid of parameters, then reducing and analyzing the results) are written as ad-hoc scripts in which the scientific logic is entangled with loops, argument plumbing, file paths, and cluster-specific submission code.
Such scripts are difficult to test, to resume after partial failure, and to migrate between a laptop and an HPC cluster.
Existing workflow tools solve parts of this problem but typically demand that the user restructure their code or operate substantial infrastructure (see *State of the field*).

`pipefunc` addresses this with a deliberately small conceptual surface: ordinary Python functions plus output names plus mapspecs.
Crucially, the function bodies and signatures remain framework-agnostic—they accept and return plain Python objects and carry no framework-specific types.
All orchestration metadata lives in a thin wrapper, applied either as the `@pipefunc` decorator or at pipeline-assembly time (`PipeFunc(func, ...)`), so existing functions can be used entirely untouched.
Code that already follows good engineering practice—small, pure functions with explicit inputs and outputs—can therefore be adopted into a pipeline without modification, remains unit-testable in isolation, and stays reusable outside the framework.
The same pipeline definition runs sequentially in a notebook, in parallel on a workstation, or on a SLURM [@yoo2003slurm] cluster, with results stored in pluggable backends (in-memory, file-based, or—via the optional Zarr [@zarr] integration—cloud-capable stores) and automatically reassembled into an `xarray.Dataset`.

`pipefunc` is aimed at researchers who think in functions rather than in workflow-engine concepts: the cost of adopting it is a decorator per function, and the cost of leaving is deleting them.
Its target audience is computational scientists—in physics, chemistry, materials science, engineering, and machine-learning research—whose daily unit of work is a multidimensional parameter sweep over expensive simulations.

# State of the field

The gap `pipefunc` targets sits between two well-served extremes.
On one side, workflow engines such as AiiDA [@huber2020aiida], Snakemake [@molder2021snakemake], Airflow [@airflow], and Luigi [@luigi] offer capabilities such as provenance tracking, scheduling, and multi-language support, but impose substantial infrastructure (databases, daemons, domain-specific languages, file-based interfaces) that is disproportionate for Python-centric research codes and hinders interactive use.
On the other side, task-graph libraries such as Dask [@rocklin2015dask] and Parsl [@babuji2019parsl] make it easy to parallelize individual function calls, but leave the structure of the workflow—the wiring between steps, the shape of parameter sweeps, the gathering of results into analyzable arrays—as imperative code the user must write and maintain.
Hamilton [@hamilton] is closest in spirit, also deriving a DAG from function signatures, but it couples the DAG to module structure and parameter naming conventions, and lacks first-class N-dimensional sweeps and HPC resource management.

These differences are structural, which is why `pipefunc` is a separate package rather than a contribution to an existing one.
The mapspec notation requires shape inference, storage layout, and scheduling to all be driven by one index algebra, whereas Dask is organized around operations on typed collections and Hamilton around a driver that assembles the DAG from modules; adding mapspec-style sweeps to either would amount to a redesign of their execution models rather than an incremental feature.
Conversely, `pipefunc` deliberately reuses what already works: NetworkX [@hagberg2008networkx] for graph algorithms, NumPy [@harris2020numpy] for array handling, `concurrent.futures` as the executor interface (so executors from Dask, `mpi4py` [@mpi4py], and others plug in directly), Xarray for labeled results, Zarr for storage, and Adaptive [@nijholt2019adaptive] and Adaptive Scheduler [@adaptive-scheduler] for adaptive sampling and SLURM execution.

# Software design

Three design decisions define `pipefunc`.

First, the DAG is derived from function signatures and output names alone: an argument named `c` is satisfied by whichever function declares `output_name="c"`.
This convention-over-configuration approach eliminates explicit wiring code, at the cost of requiring consistent naming—a cost mitigated by renaming support and parameter scopes for composing pipelines from independently developed parts.

Second, all sweep structure is declarative.
Because mapspecs are data rather than code, `pipefunc` can validate shapes before execution, parallelize element-wise operations automatically, store each element independently (enabling resumption of partially completed runs), and rewrite the sweep structure programmatically.
The trade-off is a learning curve for the index notation, which is kept minimal by mirroring NumPy's einsum syntax.

Third, execution, storage, and resource management are orthogonal plug-ins.
A pipeline runs on any `concurrent.futures.Executor`; results go to any of several storage backends; per-function resource requirements (CPUs, memory, GPUs, wall time) can be declared—optionally as functions of the input arguments—and are translated into SLURM job submissions automatically.
Different functions in one pipeline can use different executors and stores.
The per-function scheduling overhead is roughly 15 µs, so the framework remains usable even for pipelines of many fast functions.

The following condensed example, modeled on a typical device-simulation workflow, shows the pieces working together (imports and dataclass definitions are omitted; the full runnable version is the physics-based example in the documentation).
The objects passed between functions are plain frozen dataclasses; the functions contain no `pipefunc`-specific code beyond the decorator.

```python
@pipefunc(output_name="geo")
def make_geometry(x: float, y: float) -> Geometry:
    return Geometry(x, y)

@pipefunc(output_name=("mesh", "coarse_mesh"))  # multiple outputs
def make_mesh(geo: Geometry, mesh_size: float, coarse_mesh_size: float = 0.1):
    return Mesh(geo, mesh_size), Mesh(geo, coarse_mesh_size)

@pipefunc(output_name="materials")
def make_materials(geo: Geometry) -> Materials:
    return Materials(geo, ["i", "j", "c"])

@pipefunc(output_name="electrostatics", mapspec="V_left[i], V_right[j] -> electrostatics[i, j]")
def run_electrostatics(mesh, materials, V_left: float, V_right: float) -> Electrostatics:
    return Electrostatics(mesh, materials, [V_left, V_right])

@pipefunc(output_name="charge", mapspec="electrostatics[i, j] -> charge[i, j]")
def get_charge(electrostatics: Electrostatics) -> float:
    return sum(electrostatics.voltages)

@pipefunc(output_name="average_charge")  # no mapspec: receives the full 2D array
def average_charge(charge: np.ndarray) -> float:
    return np.mean(charge)

pipeline = Pipeline([make_geometry, make_mesh, make_materials,
                     run_electrostatics, get_charge, average_charge])
results = pipeline.map(inputs, run_folder="run", parallel=True)
```

The mapspec on `run_electrostatics` declares an outer product over the voltage arrays, `get_charge` maps element-wise over the resulting 2D array, and `average_charge`—having no mapspec—receives the fully assembled array and reduces it.
Extending the study requires no changes to the functions: `Pipeline.add_mapspec_axis` rewrites the mapspecs of all downstream functions, so a two-parameter study becomes a four-dimensional one in two lines:

```python
pipeline.add_mapspec_axis("x", axis="a")
pipeline.add_mapspec_axis("y", axis="b")
results = pipeline.map(new_inputs, run_folder="run", parallel=True)
ds = results.to_xarray()  # labeled 4D Dataset with coordinates V_left, V_right, x, y
```

\autoref{fig:pipeline} shows the resulting pipeline as rendered by `pipeline.visualize()`.
Because the DAG is known before execution, the same pipeline can be inspected (`pipeline.print_documentation()` renders the functions' docstrings as pipeline documentation), validated (type annotations are checked between connected functions), restructured (`pipeline.nest_funcs` merges nodes so that large intermediate objects are never serialized), profiled, cached, and resumed after partial failure—all without touching the underlying functions.

![The example pipeline after adding the `x` and `y` sweep axes, as rendered by `pipeline.visualize()`. Green dashed nodes are inputs, blue nodes are functions; each function node lists its outputs with mapspec indices and type annotations.\label{fig:pipeline}](pipeline.svg)

The implementation is fully typed, has over 99.9% line coverage from more than 1100 tests (enforced in continuous integration and tracked on Codecov), and has only three required dependencies.

# Research impact statement

`pipefunc` has been distributed on PyPI since 2023 and on conda-forge, and was developed to support large-scale physics simulations, in particular multidimensional parameter sweeps in quantum-device modeling.
As of June 2026, it has been downloaded over 185,000 times from conda-forge with PyPI adding roughly 34,000 downloads per month (per anaconda.org and pypistats.org), and an active user community exists around the project's GitHub repository (over 450 stars, eight code contributors) and Discord server.
Open-source adoption, independent of the author, spans several domains: `rbyte` [@rbyte], an open-source multimodal dataset library for spatial intelligence developed by Yaak, uses `pipefunc` in its core; the computational-methodology repository of the Soundscape Attributes Translation Project in acoustics research builds its analysis on `pipefunc` pipelines [@satp]; and the pyiron developers prototype an integration [@pipefunc-executorlib] of `pipefunc` with `executorlib` [@executorlib] for materials-science workflows on HPC systems.
Downstream tools are also being built on top of it, such as `flowfunc` [@flowfunc], a workflow runner that uses `pipefunc` as its execution engine.
The design lineage traces to `aiida-dynamic-workflows` [@aiida-dynamic-workflows], developed at Microsoft Quantum for simulating topological qubit devices; `pipefunc` generalizes that approach without the AiiDA infrastructure requirement.

# AI usage disclosure

The majority of the codebase was written by hand, predating the availability of capable agentic AI coding tools.
More recent contributions have been developed with the assistance of agentic coding tools based on frontier models from Anthropic and OpenAI (Claude Code and Codex), used under close human direction.
This manuscript was drafted with the assistance of Claude Code and reviewed with Codex, with substantial human involvement: the author directed the content, and reviewed, edited, and validated all text.
The software's architecture and design are the author's own, and all contributions—human- or AI-assisted—are validated by the project's test suite and review process.

# Acknowledgements

We thank the contributors to the `pipefunc` repository and the users who reported issues and suggested features.

# References
