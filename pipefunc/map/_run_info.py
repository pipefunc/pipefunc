from __future__ import annotations

import functools
import json
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import numpy as np

from pipefunc._utils import at_least_tuple, dump, equal_dicts, first, load
from pipefunc._version import __version__
from pipefunc.helpers import FileArray, FileValue

from ._adaptive_scheduler_slurm_executor import is_slurm_executor
from ._mapspec import MapSpec
from ._result import DirectValue
from ._shapes import (
    external_shape_from_mask,
    internal_shape_from_mask,
    map_shapes,
    shape_and_mask_from_mapspec,
    shape_is_resolved,
)
from ._storage_array._base import StorageBase, get_storage_class

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE

    from ._result import StoreType
    from ._types import ShapeDict, ShapeTuple, UserShapeDict


@dataclass(frozen=True, eq=True)
class RunInfo:
    """Information about a ``pipeline.map()`` run.

    The data in this class is immutable, except for ``resolved_shapes`` which
    is updated as new shapes are resolved.
    """

    inputs: dict[str, Any]
    defaults: dict[str, Any]
    all_output_names: set[str]
    shapes: dict[OUTPUT_TYPE, ShapeTuple]
    resolved_shapes: dict[OUTPUT_TYPE, ShapeTuple]
    internal_shapes: UserShapeDict | None
    shape_masks: dict[OUTPUT_TYPE, tuple[bool, ...]]
    run_folder: Path | None
    mapspecs_as_strings: list[str]
    storage: str | dict[OUTPUT_TYPE, str]
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
        internal_shapes: UserShapeDict | None = None,
        *,
        executor: dict[OUTPUT_TYPE, Executor] | None = None,
        storage: str | dict[OUTPUT_TYPE, str] | None,
        cleanup: bool = True,
    ) -> RunInfo:
        storage, run_folder = _resolve_storage_and_run_folder(run_folder, storage)
        internal_shapes = _construct_internal_shapes(internal_shapes, pipeline)
        if run_folder is not None:
            if cleanup:
                _cleanup_run_folder(run_folder)
            else:
                _compare_to_previous_run_info(pipeline, run_folder, inputs, internal_shapes)
        _check_inputs(pipeline, inputs)
        _maybe_inputs_to_disk_for_slurm(pipeline, inputs, run_folder, executor)
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

    def storage_class(self, output_name: OUTPUT_TYPE) -> type[StorageBase]:
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

    def init_store(self) -> dict[str, StoreType]:
        store: dict[str, StoreType] = {}
        name_mapping = {at_least_tuple(name): name for name in self.shapes}
        # Initialize StorageBase instances for each map spec output
        for mapspec in self.mapspecs:
            # `mapspec.output_names` is always tuple, even for single output
            output_name: OUTPUT_TYPE = name_mapping[mapspec.output_names]
            if mapspec.inputs:
                shape = self.resolved_shapes[output_name]
                mask = self.shape_masks[output_name]
                arrays = _init_arrays(
                    output_name,
                    shape,
                    mask,
                    self.storage_class(output_name),
                    self.run_folder,
                )
                store.update(zip(mapspec.output_names, arrays))

        # Set up paths or DirectValue for outputs not initialized as StorageBase
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
        if self.run_folder is None:
            return
        path = self.path(self.run_folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        del data["inputs"]  # Cannot serialize inputs
        del data["defaults"]  # or defaults
        # Here .relative_to to lstrip(run_folder) prefix for both input_paths and defaults_path
        # We used to *not* do this in versions <=0.86.0, see _legacy_fix
        data["input_paths"] = {
            k: str(v.relative_to(self.run_folder)) for k, v in self.input_paths.items()
        }
        data["all_output_names"] = sorted(data["all_output_names"])
        dicts_with_tuples = ["shapes", "shape_masks", "resolved_shapes"]
        if isinstance(self.storage, dict):
            dicts_with_tuples.append("storage")
        for key in dicts_with_tuples:
            data[key] = {_maybe_tuple_to_str(k): v for k, v in data[key].items()}
        data["run_folder"] = str(data["run_folder"].absolute())
        data["defaults_path"] = str(self.defaults_path.relative_to(self.run_folder))
        with path.open("w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls: type[RunInfo], run_folder: str | Path) -> RunInfo:
        path = cls.path(run_folder)  # run_info.json
        run_folder_abs = path.absolute().parent
        with path.open() as f:
            data = json.load(f)
        _legacy_fix(data, run_folder_abs)
        data["input_paths"] = {k: run_folder_abs / v for k, v in data["input_paths"].items()}
        data["all_output_names"] = set(data["all_output_names"])
        if isinstance(data["storage"], dict):
            data["storage"] = {_maybe_str_to_tuple(k): v for k, v in data["storage"].items()}
        for key in ["shapes", "shape_masks", "resolved_shapes"]:
            data[key] = {_maybe_str_to_tuple(k): tuple(v) for k, v in data[key].items()}
        if data["internal_shapes"] is not None:
            data["internal_shapes"] = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in data["internal_shapes"].items()
            }
        data["run_folder"] = run_folder_abs
        data["inputs"] = {k: load(v) for k, v in data.pop("input_paths").items()}
        data["defaults"] = load(run_folder_abs / data.pop("defaults_path"))
        return cls(**data)

    @staticmethod
    def path(run_folder: str | Path) -> Path:
        return Path(run_folder) / "run_info.json"

    def resolve_downstream_shapes(
        self,
        output_name: str,
        store: dict[str, StoreType],
        output: Any | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        if output_name not in self.resolved_shapes:
            return
        if shape_is_resolved(self.resolved_shapes[output_name]):
            return
        # After a new shape is known, update downstream shapes
        internal: ShapeDict = {
            name: internal_shape_from_mask(shape, self.shape_masks[name])
            for name, shape in self.resolved_shapes.items()
            if not isinstance(name, tuple)
        }
        if output is not None:
            assert shape is None
            shape = np.shape(output)
        assert shape is not None
        internal[output_name] = internal_shape_from_mask(shape, self.shape_masks[output_name])
        # RunInfo.mapspecs is topologically ordered
        mapspecs = {name: mapspec for mapspec in self.mapspecs for name in mapspec.output_names}
        has_updated = False
        for name, _shape in self.resolved_shapes.items():
            if not shape_is_resolved(_shape):
                mapspec = mapspecs[first(name)]
                new_shape, mask = shape_and_mask_from_mapspec(
                    mapspec,
                    self.resolved_shapes,
                    internal,
                )
                assert mask == self.shape_masks[name]
                self.resolved_shapes[name] = new_shape
                if shape_is_resolved(new_shape):
                    has_updated = True
                if not isinstance(name, tuple):
                    _update_shape_in_store(new_shape, mask, store, name)
                    internal[name] = internal_shape_from_mask(new_shape, self.shape_masks[name])
        if has_updated:
            self.dump()


def _legacy_fix(data: dict, run_folder: Path) -> None:
    """Fix legacy format where paths included run_folder prefix.

    Legacy format (<=v0.86.0):
    - run_folder: "foo/my_run_folder"
    - input_paths: {"x": "foo/my_run_folder/inputs/x.cloudpickle"}
    - defaults_path: "foo/my_run_folder/defaults/defaults.cloudpickle"

    New format (>v0.86.0):
    - run_folder: /absolute/path/to/foo/my_run_folder
    - input_paths: {"x": "inputs/x.cloudpickle"}
    - defaults_path: "defaults/defaults.cloudpickle"

    Parameters
    ----------
    data
        RunInfo data dict (modified in place)
    run_folder
        Original run_folder path

    """
    stored_run_folder = data["run_folder"]

    # Detect legacy: check if paths start with stored run_folder
    is_legacy = data["defaults_path"].startswith(stored_run_folder)

    if not is_legacy:
        return

    # Fix paths: strip the stored run_folder prefix
    stored_prefix = Path(stored_run_folder)

    data["run_folder"] = str(run_folder.resolve())
    data["defaults_path"] = str(Path(data["defaults_path"]).relative_to(stored_prefix))
    data["input_paths"] = {
        k: str(Path(v).relative_to(stored_prefix)) for k, v in data["input_paths"].items()
    }


# Max size for inputs in bytes (100 kB)
_MAX_SIZE_BYTES_INPUT = 100 * 1024


def _maybe_inputs_to_disk_for_slurm(
    pipeline: Pipeline,
    inputs: dict[str, Any],
    run_folder: Path | None,
    executor: dict[OUTPUT_TYPE, Executor] | None,
) -> None:
    """If `run_folder` is set, dump inputs to disk if large serialization required.

    Only relevant if the input is used in a SlurmExecutor.

    This automatically applies the fix described in
    https://github.com/pipefunc/pipefunc/blob/fbab121d/docs/source/concepts/slurm.md?plain=1#L263-L366
    """
    if run_folder is None:
        return
    for input_name, value in inputs.items():
        if not _input_used_in_slurm_executor(pipeline, input_name, executor):
            continue
        dumped = cloudpickle.dumps(value)
        if len(dumped) < _MAX_SIZE_BYTES_INPUT:
            continue
        warnings.warn(
            f"Input `{input_name}` is too large ({len(dumped) / 1024} kB), "
            "dumping to disk instead of serializing.",
            stacklevel=2,
        )
        input_path = _input_path(input_name, run_folder)
        path = input_path.with_suffix("")
        new_value: FileArray | FileValue
        if input_name in pipeline.mapspec_names:
            new_value = FileArray.from_data(value, path)
        else:
            new_value = FileValue.from_data(value, path)
        inputs[input_name] = new_value


def _update_shape_in_store(
    shape: ShapeTuple,
    mask: tuple[bool, ...],
    store: dict[str, StoreType],
    name: str,
) -> None:
    storage = store.get(name)
    if isinstance(storage, StorageBase):
        external_shape = external_shape_from_mask(shape, mask)
        assert len(storage.shape) == len(external_shape)
        storage.shape = external_shape


def _requires_serialization(storage: str | dict[OUTPUT_TYPE, str]) -> bool:
    if isinstance(storage, str):
        return get_storage_class(storage).requires_serialization
    return any(get_storage_class(s).requires_serialization for s in storage.values())


def _resolve_storage_and_run_folder(
    run_folder: str | Path | None,
    storage: str | dict[OUTPUT_TYPE, str] | None,
) -> tuple[str | dict[OUTPUT_TYPE, str], Path | None]:
    if run_folder is not None:
        return storage or "file_array", Path(run_folder)

    if storage is None:
        return "dict", None

    if _requires_serialization(storage):
        temp_folder = Path(tempfile.mkdtemp())
        msg = f"{storage} storage requires a `run_folder`. Using temporary folder: `{temp_folder}`."
        warnings.warn(msg, stacklevel=2)
        return storage, temp_folder

    return storage, None


def _construct_internal_shapes(
    internal_shapes: UserShapeDict | None,
    pipeline: Pipeline,
) -> UserShapeDict | None:
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


def _cleanup_run_folder(run_folder: Path) -> None:
    """Remove the run folder and its contents."""
    shutil.rmtree(run_folder, ignore_errors=True)


def _compare_to_previous_run_info(
    pipeline: Pipeline,
    run_folder: Path,
    inputs: dict[str, Any],
    internal_shapes: UserShapeDict | None = None,
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


def _init_arrays(
    output_name: OUTPUT_TYPE,
    shape: ShapeTuple,
    mask: tuple[bool, ...],
    storage_class: type[StorageBase],
    run_folder: Path | None,
) -> list[StorageBase]:
    external_shape = external_shape_from_mask(shape, mask)
    internal_shape = internal_shape_from_mask(shape, mask)
    output_names = at_least_tuple(output_name)
    paths = [_maybe_array_path(output_name, run_folder) for output_name in output_names]  # type: ignore[misc]
    return [storage_class(path, external_shape, internal_shape, mask) for path in paths]


def _maybe_array_path(output_name: str, run_folder: Path | None) -> Path | None:
    if run_folder is None:
        return None
    assert isinstance(output_name, str)
    return run_folder / "outputs" / output_name


def _input_used_in_slurm_executor(
    pipeline: Pipeline,
    input_name: str,
    executor: dict[OUTPUT_TYPE, Executor] | None,
) -> bool:
    if executor is None:
        return False
    from ._run import _executor_for_func

    dependents = pipeline.func_dependents(input_name)
    for output_name in dependents:
        func = pipeline[output_name]
        ex = _executor_for_func(func, executor)
        if is_slurm_executor(ex) and func.resources_scope == "element":
            return True
    return False
