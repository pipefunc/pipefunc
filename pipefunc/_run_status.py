"""Helpers for inspecting persisted pipeline runs from disk."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipefunc.map import load_all_outputs
from pipefunc.map._run_info import (
    RunInfo,
    _init_arrays,
    _legacy_fix,
    _maybe_str_to_tuple,
    _output_path,
)
from pipefunc.map._shapes import shape_is_resolved
from pipefunc.map._storage_array._base import StorageBase, get_storage_class


def status_from_run_folder(
    run_folder: str | Path,
    *,
    include_outputs: bool = True,
    include_run_info: bool = False,
) -> dict[str, Any]:
    """Summarize the status of a persisted run folder.

    The status is inferred from files on disk and does not query any external
    scheduler. That means incomplete runs are reported heuristically as
    ``"pending"``, ``"running"``, or ``"incomplete"``.
    """
    try:
        metadata, run_info_json, run_info_path = _load_run_metadata(run_folder)
    except Exception as e:  # noqa: BLE001
        return {
            "run_folder": str(Path(run_folder).absolute()),
            "status": "missing",
            "status_source": "disk_heuristic",
            "error": str(e),
        }

    outputs, all_complete = _progress_info_from_disk(metadata)
    progress_fraction = _overall_progress(outputs)
    status = _derive_status(all_complete=all_complete, progress_fraction=progress_fraction)

    result = {
        "run_folder": str(metadata["run_folder"]),
        "status": status,
        "status_source": "disk_heuristic",
        "all_complete": all_complete,
        "progress_fraction": progress_fraction,
        "n_outputs": len(outputs),
        "n_outputs_completed": sum(1 for output in outputs.values() if output["complete"]),
        "last_modified": _isoformat_timestamp(run_info_path.stat().st_mtime),
        "pipefunc_version": run_info_json.get("pipefunc_version", "unknown"),
    }
    if include_outputs:
        result["outputs"] = outputs
    if include_run_info:
        result["run_info"] = run_info_json
    return result


def list_run_statuses(
    folder: str | Path = "runs",
    max_runs: int | None = None,
) -> dict[str, Any]:
    """List run folders with compact status summaries."""
    runs_folder = Path(folder)
    result: dict[str, Any] = {
        "runs": [],
        "total_count": 0,
        "folder": str(runs_folder),
        "scanned_directories": 0,
        "error": None,
    }
    if not runs_folder.exists():
        result["error"] = f"Folder '{runs_folder}' does not exist"
        return result
    if not runs_folder.is_dir():
        result["error"] = f"'{runs_folder}' is not a directory"
        return result

    candidates: list[tuple[float, Path]] = []
    for run_folder in runs_folder.iterdir():
        if not run_folder.is_dir():
            continue
        result["scanned_directories"] += 1
        run_info_path = RunInfo.path(run_folder)
        if not run_info_path.exists():
            continue
        candidates.append((run_info_path.stat().st_mtime, run_folder))

    candidates.sort(key=lambda item: item[0], reverse=True)
    if max_runs is not None and max_runs > 0:
        candidates = candidates[:max_runs]

    for _mtime, run_folder in candidates:
        result["runs"].append(status_from_run_folder(run_folder, include_outputs=False))

    result["total_count"] = len(result["runs"])
    return result


def run_info(run_folder: str | Path) -> dict[str, Any]:
    """Inspect a run folder and include raw ``run_info.json`` content."""
    status = status_from_run_folder(
        run_folder,
        include_outputs=True,
        include_run_info=True,
    )
    if "error" in status:
        return {"error": status["error"]}
    return {
        "run_info": status["run_info"],
        "outputs": status["outputs"],
        "all_complete": status["all_complete"],
    }


def list_historical_runs(
    folder: str | Path = "runs",
    max_runs: int | None = None,
) -> dict[str, Any]:
    """List historical run folders with a compact compatibility schema."""
    statuses = list_run_statuses(folder=folder, max_runs=max_runs)
    if statuses["error"] is not None:
        return statuses

    runs = []
    for run in statuses["runs"]:
        if "error" in run:
            continue
        runs.append(
            {
                "last_modified": run["last_modified"],
                "run_folder": run["run_folder"],
                "all_complete": run["all_complete"],
                "total_outputs": run["n_outputs"],
                "completed_outputs": run["n_outputs_completed"],
                "pipefunc_version": run["pipefunc_version"],
            },
        )

    return {
        "runs": runs,
        "total_count": len(runs),
        "folder": statuses["folder"],
        "scanned_directories": statuses["scanned_directories"],
        "error": None,
    }


def load_outputs(run_folder: str | Path, output_names: list[str] | None = None) -> dict[str, Any]:
    """Load outputs from a persisted run folder."""
    try:
        result = load_all_outputs(run_folder=run_folder)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    if output_names is not None:
        return {key: result[key] for key in output_names}
    return result


def _load_run_metadata(run_folder: str | Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_folder_path = Path(run_folder).absolute()
    run_info_path = RunInfo.path(run_folder_path)
    if not run_info_path.exists():
        msg = f"Run folder '{run_folder_path}' does not contain run_info.json"
        raise FileNotFoundError(msg)

    run_info_json = json.loads(run_info_path.read_text())
    _legacy_fix(run_info_json, run_info_path.absolute().parent)
    metadata = {
        "all_output_names": sorted(run_info_json["all_output_names"]),
        "run_folder": Path(run_info_json["run_folder"]),
        "storage": _deserialize_storage(run_info_json["storage"]),
        "resolved_shapes": _deserialize_mapping(run_info_json["resolved_shapes"]),
        "shape_masks": _deserialize_mapping(run_info_json["shape_masks"]),
    }
    return metadata, run_info_json, run_info_path


def _deserialize_mapping(data: dict[str, list[int] | list[bool]]) -> dict[Any, tuple[Any, ...]]:
    return {_maybe_str_to_tuple(key): tuple(value) for key, value in data.items()}


def _deserialize_storage(storage: str | dict[str, str]) -> str | dict[Any, str]:
    if isinstance(storage, str):
        return storage
    return {_maybe_str_to_tuple(key): value for key, value in storage.items()}


def _progress_info_from_disk(metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    outputs: dict[str, Any] = {}
    all_complete = True
    for output_name in metadata["all_output_names"]:
        output_status = _output_status(output_name, metadata)
        outputs[output_name] = output_status
        all_complete = all_complete and output_status["complete"]
    return outputs, all_complete


def _output_status(output_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    shape_info = _shape_and_mask_for_output(
        output_name,
        resolved_shapes=metadata["resolved_shapes"],
        shape_masks=metadata["shape_masks"],
    )
    if shape_info is None:
        path = _output_path(output_name, metadata["run_folder"])
        if path.exists():
            return {"progress": 1.0, "complete": True, "bytes": path.stat().st_size}
        return {"progress": 0.0, "complete": False, "bytes": 0}

    shape, mask = shape_info
    storage_name = _storage_name_for_output(metadata["storage"], output_name)
    storage_class = get_storage_class(storage_name)
    if not storage_class.requires_serialization:
        return {"progress": "unknown", "complete": False, "bytes": 0}

    storage = _init_arrays(
        output_name,
        shape,
        mask,
        storage_class,
        metadata["run_folder"],
    )[0]
    if not isinstance(storage, StorageBase):  # pragma: no cover
        return {"progress": "unknown", "complete": False, "bytes": 0}

    progress: float | str
    if shape_is_resolved(storage.shape):
        size = storage.size
        progress = 1.0 if size == 0 else 1.0 - sum(storage.mask_linear()) / size
    else:
        progress = "unknown"

    return {
        "progress": progress,
        "complete": progress == 1.0,
        "bytes": _storage_bytes(storage),
    }


def _shape_and_mask_for_output(
    output_name: str,
    *,
    resolved_shapes: dict[Any, tuple[Any, ...]],
    shape_masks: dict[Any, tuple[Any, ...]],
) -> tuple[tuple[Any, ...], tuple[Any, ...]] | None:
    for key, shape in resolved_shapes.items():
        if key == output_name or (isinstance(key, tuple) and output_name in key):
            return shape, shape_masks[key]
    return None


def _storage_name_for_output(storage: str | dict[Any, str], output_name: str) -> str:
    if isinstance(storage, str):
        return storage
    default = storage.get("")
    storage_name = storage.get(output_name, default)
    if storage_name is None:
        msg = (
            f"Cannot find storage class for '{output_name}'. "
            f"Available keys: {sorted(map(str, storage))}"
        )
        raise ValueError(msg)
    return storage_name


def _storage_bytes(storage: StorageBase) -> int:
    folder = getattr(storage, "folder", None)
    if folder is None:
        return 0
    return sum(path.stat().st_size for path in Path(folder).rglob("*") if path.is_file())


def _overall_progress(outputs: dict[str, Any]) -> float | None:
    values = [
        float(output["progress"])
        for output in outputs.values()
        if isinstance(output["progress"], int | float)
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _derive_status(*, all_complete: bool, progress_fraction: float | None) -> str:
    if all_complete:
        return "completed"
    if progress_fraction is None:
        return "incomplete"
    if progress_fraction == 0.0:
        return "pending"
    return "running"


def _isoformat_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
