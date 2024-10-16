from __future__ import annotations

import functools
import json
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, TypeVar

from pipefunc._utils import at_least_tuple, dump, equal_dicts, load
from pipefunc._version import __version__
from pipefunc.map._mapspec import MapSpec, array_shape
from pipefunc.map._safe_eval import evaluate_expression
from pipefunc.map._storage_base import StorageBase, get_storage_class

if TYPE_CHECKING:
    from pipefunc import Pipeline

_OUTPUT_TYPE: TypeAlias = str | tuple[str, ...]


class _Missing: ...


class DirectValue:
    __slots__ = ["value"]

    def __init__(self, value: Any | type[_Missing] = _Missing) -> None:
        self.value = value

    def exists(self) -> bool:
        return self.value is not _Missing


class Shapes(NamedTuple):
    shapes: dict[_OUTPUT_TYPE, tuple[int | str, ...]]
    masks: dict[_OUTPUT_TYPE, tuple[bool, ...]]


def _input_shapes_and_masks(
    pipeline: Pipeline,
    inputs: dict[str, Any],
) -> Shapes:
    input_parameters = set(pipeline.topological_generations.root_args)
    inputs_with_defaults = pipeline.defaults | inputs
    shapes: dict[_OUTPUT_TYPE, tuple[int | str, ...]] = {  # In this function, we only use ints
        p: array_shape(inputs_with_defaults[p], p)
        for p in input_parameters
        if p in pipeline.mapspec_names
    }
    masks = {name: len(shape) * (True,) for name, shape in shapes.items()}
    return Shapes(shapes, masks)


def map_shapes(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    internal_shapes: dict[str, int | str | tuple[int | str, ...]] | None = None,
) -> Shapes:
    if internal_shapes is None:
        internal_shapes = {}
    internal = {k: at_least_tuple(v) for k, v in internal_shapes.items()}
    shapes: dict[_OUTPUT_TYPE, tuple[int | str, ...]] = {}
    masks: dict[_OUTPUT_TYPE, tuple[bool, ...]] = {}
    input_shapes, input_masks = _input_shapes_and_masks(pipeline, inputs)
    shapes.update(input_shapes)
    masks.update(input_masks)

    mapspec_funcs = [f for f in pipeline.sorted_functions if f.mapspec]
    for func in mapspec_funcs:
        assert func.mapspec is not None  # mypy
        output_shape, mask = _shape_and_mask(func.mapspec, shapes, internal)
        shapes[func.output_name] = output_shape
        masks[func.output_name] = mask
        if isinstance(func.output_name, tuple):
            for output_name in func.output_name:
                shapes[output_name] = output_shape
                masks[output_name] = mask

    assert all(k in shapes for k in pipeline.mapspec_names if k not in internal)
    return Shapes(shapes, masks)


def _shape_and_mask(
    mapspec: MapSpec,
    shapes: dict[_OUTPUT_TYPE, tuple[int | str, ...]],
    internal_shapes: dict[str, tuple[int | str, ...]],
) -> tuple[tuple[int | str, ...], tuple[bool, ...]]:
    input_shapes = {p: shapes[p] for p in mapspec.input_names if p in shapes}
    output_shapes = {p: internal_shapes[p] for p in mapspec.output_names if p in internal_shapes}
    output_shape, mask = mapspec.shape(input_shapes, output_shapes)  # type: ignore[arg-type]
    return output_shape, mask


@dataclass(frozen=True, eq=True)
class RunInfo:
    inputs: dict[str, Any]
    defaults: dict[str, Any]
    all_output_names: set[str]
    shapes: dict[_OUTPUT_TYPE, tuple[int | str, ...]]
    resolved_shapes: dict[_OUTPUT_TYPE, tuple[int | str, ...]]
    internal_shapes: dict[str, int | str | tuple[int | str, ...]] | None
    shape_masks: dict[_OUTPUT_TYPE, tuple[bool, ...]]
    run_folder: Path | None
    mapspecs_as_strings: list[str]
    storage: str | dict[_OUTPUT_TYPE, str]
    pipefunc_version: str = __version__

    def __post_init__(self) -> None:
        if self.run_folder is None:
            return
        self.dump()
        for input_name, value in self.inputs.items():
            input_path = _input_path(input_name, self.run_folder)
            dump(value, input_path)
        defaults_path = _defaults_path(self.run_folder)
        dump(self.defaults, defaults_path)

    @classmethod
    def create(
        cls: type[RunInfo],
        run_folder: str | Path | None,
        pipeline: Pipeline,
        inputs: dict[str, Any],
        internal_shapes: dict[str, int | str | tuple[int | str, ...]] | None = None,
        *,
        storage: str | dict[_OUTPUT_TYPE, str],
        cleanup: bool = True,
    ) -> RunInfo:
        run_folder = _maybe_run_folder(run_folder, storage)
        if run_folder is not None:
            if cleanup:
                _cleanup_run_folder(run_folder)
            else:
                _compare_to_previous_run_info(pipeline, run_folder, inputs, internal_shapes)
        _check_inputs(pipeline, inputs)
        internal_shapes = _construct_internal_shapes(internal_shapes, pipeline)
        shapes, masks = map_shapes(pipeline, inputs, internal_shapes)
        resolved_shapes = shapes.copy()
        return cls(
            inputs=inputs,
            defaults=pipeline.defaults,
            all_output_names=pipeline.all_output_names,
            shapes=shapes,
            resolved_shapes=resolved_shapes,
            internal_shapes=internal_shapes,
            shape_masks=masks,
            mapspecs_as_strings=pipeline.mapspecs_as_strings,
            run_folder=run_folder,
            storage=storage,
        )

    def storage_class(self, output_name: _OUTPUT_TYPE) -> type[StorageBase]:
        if isinstance(self.storage, str):
            return get_storage_class(self.storage)
        default: str | None = self.storage.get("")
        storage: str | None = self.storage.get(output_name, default)
        if storage is None:
            msg = (
                f"Cannot find storage class for `{output_name}`."
                f" Either add `storage[{output_name}] = ...` or"
                ' use a default by setting `storage[""] = ...`.'
            )
            raise ValueError(msg)
        return get_storage_class(storage)

    def init_store(self) -> dict[str, LazyStorage | StorageBase | Path | DirectValue]:
        store: dict[str, LazyStorage | StorageBase | Path | DirectValue] = {}
        name_mapping = {at_least_tuple(name): name for name in self.shapes}
        # Initialize LazyStore instances for each map spec output
        for mapspec in self.mapspecs:
            # `mapspec.output_names` is always tuple, even for single output
            output_name: _OUTPUT_TYPE = name_mapping[mapspec.output_names]
            if mapspec.inputs:
                shape = self.shapes[output_name]
                mask = self.shape_masks[output_name]
                storages = _init_storages(
                    output_name,
                    shape,
                    mask,
                    self.storage_class(output_name),
                    self.run_folder,
                )
                store.update(zip(mapspec.output_names, storages))

        # Set up paths or DirectValue for outputs not initialized as LazyStore
        for output_name in self.all_output_names:
            if output_name not in store:
                store[output_name] = (
                    _output_path(output_name, self.run_folder)
                    if isinstance(self.run_folder, Path)
                    else DirectValue()
                )
        return store

    @property
    def input_paths(self) -> dict[str, Path]:
        if self.run_folder is None:  # pragma: no cover
            msg = "Cannot get `input_paths` without `run_folder`."
            raise ValueError(msg)
        return {k: _input_path(k, self.run_folder) for k in self.inputs}

    @property
    def defaults_path(self) -> Path:
        if self.run_folder is None:  # pragma: no cover
            msg = "Cannot get `defaults_path` without `run_folder`."
            raise ValueError(msg)
        return _defaults_path(self.run_folder)

    @functools.cached_property
    def mapspecs(self) -> list[MapSpec]:
        return [MapSpec.from_string(ms) for ms in self.mapspecs_as_strings]

    def dump(self) -> None:
        """Dump the RunInfo to a file."""
        if self.run_folder is None:  # pragma: no cover
            msg = "Cannot dump `RunInfo` without `run_folder`."
            raise ValueError(msg)
        path = self.path(self.run_folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        del data["inputs"]  # Cannot serialize inputs
        del data["defaults"]  # or defaults
        del data["resolved_shapes"]
        data["input_paths"] = {k: str(v) for k, v in self.input_paths.items()}
        data["all_output_names"] = sorted(data["all_output_names"])
        dicts_with_tuples = ["shapes", "shape_masks"]
        if isinstance(self.storage, dict):
            dicts_with_tuples.append("storage")
        for key in dicts_with_tuples:
            data[key] = {_maybe_tuple_to_str(k): v for k, v in data[key].items()}
        data["run_folder"] = str(data["run_folder"])
        data["defaults_path"] = str(self.defaults_path)
        with path.open("w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls: type[RunInfo], run_folder: str | Path) -> RunInfo:
        path = cls.path(run_folder)
        with path.open() as f:
            data = json.load(f)
        data["input_paths"] = {k: Path(v) for k, v in data["input_paths"].items()}
        data["all_output_names"] = set(data["all_output_names"])
        if isinstance(data["storage"], dict):
            data["storage"] = {_maybe_str_to_tuple(k): v for k, v in data["storage"].items()}
        for key in ["shapes", "shape_masks"]:
            data[key] = {_maybe_str_to_tuple(k): tuple(v) for k, v in data[key].items()}
        if data["internal_shapes"] is not None:
            data["internal_shapes"] = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in data["internal_shapes"].items()
            }
        data["run_folder"] = Path(data["run_folder"])
        data["inputs"] = {k: load(Path(v)) for k, v in data.pop("input_paths").items()}
        data["defaults"] = load(Path(data.pop("defaults_path")))
        data["resolved_shapes"] = data["shapes"]
        return cls(**data)

    @staticmethod
    def path(run_folder: str | Path) -> Path:
        return Path(run_folder) / "run_info.json"

    def resolve_shapes(self, output_name: _OUTPUT_TYPE, kwargs: dict[str, Any]) -> None:
        """Resolve shapes that depend on kwargs."""
        if output_name not in self.resolved_shapes:
            return
        shape = self.resolved_shapes[output_name]
        if any(isinstance(i, str) for i in shape):
            new_shape = _resolve_shape(shape, kwargs)
            self.resolved_shapes[output_name] = new_shape
            for name in at_least_tuple(output_name):
                self.resolved_shapes[name] = new_shape
        else:
            return

        # Now that new shape is known, update downstream shapes
        internal = {
            name: _internal_shape(shape, self.shape_masks[name])
            for name, shape in self.resolved_shapes.items()
        }

        mapspecs = {name: mapspec for mapspec in self.mapspecs for name in mapspec.output_names}
        for name, shape in self.resolved_shapes.items():
            if any(isinstance(i, str) for i in shape):
                new_shape, _ = _shape_and_mask(mapspecs[name], self.resolved_shapes, internal)
                self.resolved_shapes[name] = new_shape
                internal[name] = _internal_shape(shape, self.shape_masks[name])


@dataclass
class LazyStorage:
    """Object that can generate a StorageBase instance on demand."""

    output_name: str
    shape: tuple[int | str, ...]
    shape_mask: tuple[bool, ...]
    storage_class: type[StorageBase]
    run_folder: Path | None

    def evaluate(self, kwargs: dict[str, Any] | None = None) -> StorageBase:
        if kwargs is None:
            kwargs = {}
        shape: tuple[int, ...] = _resolve_shape(self.shape, kwargs)
        path = _maybe_file_array_path(self.output_name, self.run_folder)
        external_shape = _external_shape(shape, self.shape_mask)
        internal_shape = _internal_shape(shape, self.shape_mask)
        return self.storage_class(path, external_shape, internal_shape, self.shape_mask)

    def try_evaluate(self, kwargs: dict[str, Any]) -> StorageBase:
        try:
            return self.evaluate(kwargs)
        except Exception as e:
            msg = (
                f"Error evaluating lazy store for `{self.output_name}`."
                f" The error was: `{e}`."
                f" kwargs: `{kwargs}`"
            )
            raise RuntimeError(msg) from e

    def maybe_evaluate(self) -> StorageBase | LazyStorage:
        if all(isinstance(i, int) for i in self.shape):
            return self.evaluate()
        return self


def _resolve_shape(
    shape: tuple[int | str, ...],
    kwargs: dict[str, Any],
) -> tuple[int, ...]:
    resolved_shape: list[int] = []
    for x in shape:
        if isinstance(x, int):
            resolved_shape.append(x)
        else:
            i = evaluate_expression(x, kwargs)
            if not isinstance(i, int):
                msg = f"Expression `{x}` must evaluate to an integer but it evaluated to `{i}`."
                raise TypeError(msg)
            resolved_shape.append(i)
    return tuple(resolved_shape)


def _requires_serialization(storage: str | dict[_OUTPUT_TYPE, str]) -> bool:
    if isinstance(storage, str):
        return get_storage_class(storage).requires_serialization
    return any(get_storage_class(s).requires_serialization for s in storage.values())


def _maybe_run_folder(
    run_folder: str | Path | None,
    storage: str | dict[_OUTPUT_TYPE, str],
) -> Path | None:
    if run_folder is None and _requires_serialization(storage):
        run_folder = tempfile.mkdtemp()
        msg = f"{storage} storage requires a `run_folder`. Using temporary folder: `{run_folder}`."
        warnings.warn(msg, stacklevel=2)
    return Path(run_folder) if run_folder is not None else None


# TODO: remove and make `internal_shapes` a property of RunInfo
def _construct_internal_shapes(
    internal_shapes: dict[str, int | str | tuple[int | str, ...]] | None,
    pipeline: Pipeline,
) -> dict[str, int | str | tuple[int | str, ...]] | None:
    if internal_shapes is None:
        internal_shapes = {}
    for f in pipeline.functions:
        if f.output_name in internal_shapes:
            continue
        if f.internal_shape is None:
            continue
        for output_name in at_least_tuple(f.output_name):
            internal_shapes[output_name] = f.internal_shape
    if not internal_shapes:
        return None
    return internal_shapes


def _cleanup_run_folder(run_folder: str | Path) -> None:
    """Remove the run folder and its contents."""
    run_folder = Path(run_folder)
    shutil.rmtree(run_folder, ignore_errors=True)


def _compare_to_previous_run_info(
    pipeline: Pipeline,
    run_folder: Path,
    inputs: dict[str, Any],
    internal_shapes: dict[str, int | str | tuple[int | str, ...]] | None = None,
) -> None:  # pragma: no cover
    if not RunInfo.path(run_folder).is_file():
        return
    try:
        old = RunInfo.load(run_folder)
    except Exception as e:  # noqa: BLE001
        msg = f"Could not load previous run info: {e}, cannot use `cleanup=False`."
        raise ValueError(msg) from None
    if internal_shapes != old.internal_shapes:
        msg = "Internal shapes do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    if pipeline.mapspecs_as_strings != old.mapspecs_as_strings:
        msg = "`MapSpec`s do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    shapes, _masks = map_shapes(pipeline, inputs, internal_shapes)
    if shapes != old.shapes:
        msg = "Shapes do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    equal_inputs = equal_dicts(inputs, old.inputs, verbose=True)
    if equal_inputs is None:
        print(
            "Could not compare new `inputs` to `inputs` from previous run."
            " Proceeding *without* `cleanup`, hoping for the best.",
        )
        return
    if not equal_inputs:
        msg = f"Inputs `{inputs=}` / `{old.inputs=}` do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)
    equal_defaults = equal_dicts(pipeline.defaults, old.defaults)
    if equal_defaults is None:
        print(
            "Could not compare new `defaults` to `defaults` from previous run."
            " Proceeding *without* `cleanup`, hoping for the best.",
        )
        return
    if not equal_defaults:
        msg = f"Defaults `{pipeline.defaults=}` / `{old.defaults=}` do not match previous run, cannot use `cleanup=False`."
        raise ValueError(msg)


def _check_inputs(pipeline: Pipeline, inputs: dict[str, Any]) -> None:
    input_dimensions = pipeline.mapspec_dimensions
    for name, value in inputs.items():
        if (dim := input_dimensions.get(name, 0)) > 1 and isinstance(value, list | tuple):
            msg = f"Expected {dim}D `numpy.ndarray` for input `{name}`, got {type(value)}."
            raise ValueError(msg)


def _maybe_str_to_tuple(x: str) -> tuple[str, ...] | str:
    if "," in x:
        return tuple(x.split(","))
    return x


def _maybe_tuple_to_str(x: tuple[str, ...] | str) -> str:
    if isinstance(x, tuple):
        return ",".join(x)
    return x


def _output_path(output_name: str, run_folder: Path) -> Path:
    return run_folder / "outputs" / f"{output_name}.cloudpickle"


def _input_path(input_name: str, run_folder: Path) -> Path:
    return run_folder / "inputs" / f"{input_name}.cloudpickle"


def _defaults_path(run_folder: Path) -> Path:
    return run_folder / "defaults" / "defaults.cloudpickle"


def _init_storages(
    output_name: _OUTPUT_TYPE,
    shape: tuple[int | str, ...],
    mask: tuple[bool, ...],
    storage_class: type[StorageBase],
    run_folder: Path | None,
) -> list[StorageBase | LazyStorage]:
    return [
        LazyStorage(name, shape, mask, storage_class, run_folder).maybe_evaluate()
        for name in at_least_tuple(output_name)
    ]


def _maybe_file_array_path(output_name: str, run_folder: Path | None) -> Path | None:
    if run_folder is None:
        return None
    assert isinstance(output_name, str)
    return run_folder / "outputs" / output_name


S = TypeVar("S", str, int)


def _internal_shape(shape: tuple[S, ...], mask: tuple[bool, ...]) -> tuple[S, ...]:
    return tuple(s for s, m in zip(shape, mask) if not m)


def _external_shape(shape: tuple[S, ...], mask: tuple[bool, ...]) -> tuple[S, ...]:
    return tuple(s for s, m in zip(shape, mask) if m)
