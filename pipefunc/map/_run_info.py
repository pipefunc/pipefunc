from __future__ import annotations

import functools
import json
import shutil
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipefunc._utils import at_least_tuple, dump, equal_dicts, load
from pipefunc._version import __version__

from ._mapspec import MapSpec
from ._result import DirectValue
from ._shapes import external_shape_from_mask, internal_shape_from_mask, map_shapes
from ._storage_array._base import StorageBase, get_storage_class

if TYPE_CHECKING:
    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE

    from ._result import StoreType
    from ._types import ShapeTuple, UserShapeDict


@dataclass(frozen=True, eq=True)
class RunInfo:
    inputs: dict[str, Any]
    defaults: dict[str, Any]
    all_output_names: set[str]
    shapes: dict[OUTPUT_TYPE, ShapeTuple]
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
        storage: str | dict[OUTPUT_TYPE, str],
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
        return cls(
            inputs=inputs,
            defaults=pipeline.defaults,
            all_output_names=pipeline.all_output_names,
            shapes=shapes,
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
                shape = self.shapes[output_name]
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
        if self.run_folder is None:  # pragma: no cover
            msg = "Cannot dump `RunInfo` without `run_folder`."
            raise ValueError(msg)
        path = self.path(self.run_folder)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        del data["inputs"]  # Cannot serialize inputs
        del data["defaults"]  # or defaults
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
        return cls(**data)

    @staticmethod
    def path(run_folder: str | Path) -> Path:
        return Path(run_folder) / "run_info.json"


def _requires_serialization(storage: str | dict[OUTPUT_TYPE, str]) -> bool:
    if isinstance(storage, str):
        return get_storage_class(storage).requires_serialization
    return any(get_storage_class(s).requires_serialization for s in storage.values())


def _maybe_run_folder(
    run_folder: str | Path | None,
    storage: str | dict[OUTPUT_TYPE, str],
) -> Path | None:
    if run_folder is None and _requires_serialization(storage):
        run_folder = tempfile.mkdtemp()
        msg = f"{storage} storage requires a `run_folder`. Using temporary folder: `{run_folder}`."
        warnings.warn(msg, stacklevel=2)
    return Path(run_folder) if run_folder is not None else None


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


def _cleanup_run_folder(run_folder: str | Path) -> None:
    """Remove the run folder and its contents."""
    run_folder = Path(run_folder)
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
